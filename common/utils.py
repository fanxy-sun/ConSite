import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import (
    LambdaLR
)
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
import logging
import math
import warnings
import yaml
import random
import os
import gc
from functools import partial
from pandarallel import pandarallel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef, multilabel_confusion_matrix
from sklearn.manifold import TSNE
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, TransformerConv
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_mean
from collections import defaultdict
from tqdm import tqdm
import csv
import pandas as pd
import json
from torch.cuda.amp import GradScaler, autocast



class SharedEncoder(nn.Module):
    """
    shared encoder for aligning ESM and MSA features.
    Processes both feature types through shared parameters to encourage alignment in the embedding space.
    """
    
    def __init__(
        self,
        esm_dim: int,
        msa_dim: int,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        # Projections to common dimension
        self.esm_projection = nn.Linear(esm_dim, hidden_dim)
        self.msa_projection = nn.Linear(msa_dim, hidden_dim)
        
        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Optional cross-attention module to allow features to attend to each other
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.esm_norm = nn.LayerNorm(hidden_dim)
            self.msa_norm = nn.LayerNorm(hidden_dim)
            
        # Output projections
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    


    def forward(self, esm_features, msa_features, esm_valid_tokens_mask=None, msa_valid_tokens_mask=None):
        """
        Process both ESM and MSA features through the shared encoder.
        
        Args:
            esm_features: ESM embeddings [batch_size, seq_len, esm_dim]
            msa_features: MSA embeddings [batch_size, seq_len, msa_dim]
            esm_valid_tokens_mask: Mask for valid tokens in ESM [batch_size, seq_len]，1 表示该位置要被屏蔽（ignore），不参与注意力计算
            msa_valid_tokens_mask: Mask for valid tokens in MSA [batch_size, seq_len]，1 表示该位置要被屏蔽（ignore），不参与注意力计算
            
        Returns:
            Tuple of (encoded_esm, encoded_msa)
        """
        # Project to common dimension
        esm_proj = self.esm_projection(esm_features)
        msa_proj = self.msa_projection(msa_features)
        
        # Process through shared transformer
        esm_encoded = self.transformer(esm_proj, src_key_padding_mask=esm_valid_tokens_mask)
        msa_encoded = self.transformer(msa_proj, src_key_padding_mask=msa_valid_tokens_mask)
        
        # Optional cross-attention to allow features to attend to each other
        if self.use_cross_attention:
            # ESM attends to MSA
            esm_cross, _ = self.cross_attention(
                query=esm_encoded,
                key=msa_encoded,
                value=msa_encoded
            )
            
            # MSA attends to ESM
            msa_cross, _ = self.cross_attention(
                query=msa_encoded,
                key=esm_encoded,
                value=esm_encoded
            )
            
            # Residual connection
            esm_encoded = esm_encoded + esm_cross
            msa_encoded = msa_encoded + msa_cross
        
        # Layer normalization
        if self.use_layer_norm:
            esm_encoded = self.esm_norm(esm_encoded)
            msa_encoded = self.msa_norm(msa_encoded)
        
        # Output projection
        esm_encoded = self.output_projection(esm_encoded)
        msa_encoded = self.output_projection(msa_encoded)
        
        return esm_encoded, msa_encoded



def residue_level_soft_label_alignment(
    esm_embeddings: torch.Tensor,
    msa_embeddings: torch.Tensor,
    esm_mask: Optional[torch.Tensor] = None,
    msa_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.5,
    loss_weight: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute residue-level soft-label alignment loss between ESM and MSA embeddings.
    
    This function now has enhanced numerical stability features. When use_stable_computation=True,
    it uses the stable version with comprehensive NaN/Inf checking and handling.
    
    MSA的输出特征和ESM的输出特征互为指导

    * esm_sim 矩阵 (ESM 内部相似度)衡量的是 ESM 内部残基之间的相似度。
        esm_sim = [
            [sim(ESM_残基1, ESM_残基1), sim(ESM_残基1, ESM_残基2), sim(ESM_残基1, ESM_残基3)],
            [sim(ESM_残基2, ESM_残基1), sim(ESM_残基2, ESM_残基2), sim(ESM_残基2, ESM_残基3)],
            [sim(ESM_残基3, ESM_残基1), sim(ESM_残基3, ESM_残基2), sim(ESM_残基3, ESM_残基3)]
        ]    
    * msa_sim 矩阵 (MSA 内部相似度)衡量的是 MSA 内部残基之间的相似度。
        msa_sim = [
            [sim(MSA_残基1, MSA_残基1), sim(MSA_残基1, MSA_残基2), sim(MSA_残基1, MSA_残基3)],
            [sim(MSA_残基2, MSA_残基1), sim(MSA_残基2, MSA_残基2), sim(MSA_残基2, MSA_残基3)],
            [sim(MSA_残基3, MSA_残基1), sim(MSA_残基3, MSA_残基2), sim(MSA_残基3, MSA_残基3)]
        ]
    * cross_sim_s2m 矩阵 (ESM 到 MSA 的交叉相似度)
        cross_sim_s2m = [
            [sim(ESM_残基1, MSA_残基1), sim(ESM_残基1, MSA_残基2), sim(ESM_残基1, MSA_残基3)],
            [sim(ESM_残基2, MSA_残基1), sim(ESM_残基2, MSA_残基2), sim(ESM_残基2, MSA_残基3)],
            [sim(ESM_残基3, MSA_残基1), sim(ESM_残基3, MSA_残基2), sim(ESM_残基3, MSA_残基3)]
        ]
    * cross_sim_m2s 矩阵 (MSA 到 ESM 的交叉相似度)
        cross_sim_m2s = [
            [sim(MSA_残基1, ESM_残基1), sim(MSA_残基1, ESM_残基2), sim(MSA_残基1, ESM_残基3)],
            [sim(MSA_残基2, ESM_残基1), sim(MSA_残基2, ESM_残基2), sim(MSA_残基2, ESM_残基3)],
            [sim(MSA_残基3, ESM_残基1), sim(MSA_残基3, ESM_残基2), sim(MSA_残基3, ESM_残基3)]
        ]
    * 总结在相同的索引位置 [1, 2] 上：
        esm_sim[1, 2] 关注的是 ESM 内部的 残基1 和 残基2。
        msa_sim[1, 2] 关注的是 MSA 内部的 残基1 和 残基2。
        cross_sim_s2m[1, 2] 关注的是 ESM 的 残基1 和 MSA 的 残基2。
        cross_sim_m2s[1, 2] 关注的是 MSA 的 残基1 和 ESM 的 残基2。

    Args:
        esm_embeddings: ESM residue embeddings [B, L, D].
        msa_embeddings: MSA residue embeddings [B, L, D].
        esm_mask: Boolean mask for valid ESM tokens [B, L], where True indicates a valid token.
        msa_mask: Boolean mask for valid MSA tokens [B, L], where True indicates a valid token.
        temperature: Temperature for softmax scaling.
        loss_weight: Weighting factor for the final loss.
        eps: Small epsilon for numerical stability, especially for log and normalization.

    Returns:
        A scalar tensor representing the final alignment loss.
    """
    # --- 1. Input Validation and Pre-computation ---
    if esm_mask is None or msa_mask is None:
        raise ValueError("Both esm_mask and msa_mask must be provided.")

    # A more stable way to normalize, preventing division by zero without altering vector directions.
    esm_norm = esm_embeddings / esm_embeddings.norm(dim=-1, keepdim=True).clamp(min=eps)
    msa_norm = msa_embeddings / msa_embeddings.norm(dim=-1, keepdim=True).clamp(min=eps)

    # Create 3D pair masks [B, L, L] using broadcasting.
    esm_pair_mask = esm_mask.unsqueeze(2) & esm_mask.unsqueeze(1)
    msa_pair_mask = msa_mask.unsqueeze(2) & msa_mask.unsqueeze(1)
    cross_pair_mask = esm_mask.unsqueeze(2) & msa_mask.unsqueeze(1)

    # --- 2. Similarity and Softmax Distribution Calculation (Vectorized) ---
    esm_sim = torch.bmm(esm_norm, esm_norm.transpose(1, 2))
    msa_sim = torch.bmm(msa_norm, msa_norm.transpose(1, 2))
    cross_sim_s2m = torch.bmm(esm_norm, msa_norm.transpose(1, 2))
    cross_sim_m2s = torch.bmm(msa_norm, esm_norm.transpose(1, 2))
    
    # Use a large negative value for masking, which is standard for preventing attention to padded tokens.
    fill_value = torch.finfo(esm_sim.dtype).min
    esm_sim = esm_sim.div(temperature).masked_fill(~esm_pair_mask, fill_value)
    msa_sim = msa_sim.div(temperature).masked_fill(~msa_pair_mask, fill_value)
    cross_sim_s2m = cross_sim_s2m.div(temperature).masked_fill(~cross_pair_mask, fill_value)
    cross_sim_m2s = cross_sim_m2s.div(temperature).masked_fill(~cross_pair_mask.transpose(1, 2), fill_value)

    P_s2s = F.softmax(esm_sim, dim=-1)
    P_m2m = F.softmax(msa_sim, dim=-1)
    Q_s2m = F.softmax(cross_sim_s2m, dim=-1)
    Q_m2s = F.softmax(cross_sim_m2s, dim=-1)

    # --- 3. KL Divergence Calculation with Correct Normalization (Vectorized) ---
    def _calculate_correctly_normalized_kl(p_dist, q_dist, valid_row_mask, pair_mask):
        """
        Calculates KL divergence with per-row, per-sequence, and per-batch averaging.
        This semantically matches the original iterative implementation.
        """
        # Compute log_q safely for KL divergence.
        log_q = torch.log(q_dist.clamp(min=eps))

        # Calculate element-wise KL divergence: p * (log(p) - log(q)).
        kl_elements = F.kl_div(log_q, p_dist.clamp(min=eps), reduction='none')

        # Mask out invalid pair contributions to prevent them from affecting the sum.
        kl_elements = kl_elements.masked_fill(~pair_mask, 0.0)

        # Sum over columns to get per-row KL sum. Shape: [B, L]
        row_kl_sum = kl_elements.sum(dim=-1)

        # Count valid columns for each row to perform per-row averaging.
        # .clamp(min=1) prevents division by zero for rows that are masked out.
        valid_cols_per_row = pair_mask.sum(dim=-1).clamp(min=1).to(row_kl_sum.dtype)
        
        # Per-row average KL divergence.
        row_kl_avg = row_kl_sum / valid_cols_per_row

        # Mask out invalid rows entirely so they don't contribute to the sequence sum.
        row_kl_avg = row_kl_avg.masked_fill(~valid_row_mask, 0.0)
        
        # Sum the averaged row KLs for each sequence.
        per_seq_kl_sum = row_kl_avg.sum(dim=-1)
        
        # Count valid rows for each sequence to perform per-sequence averaging.
        valid_rows_per_seq = valid_row_mask.sum(dim=-1).clamp(min=1).to(per_seq_kl_sum.dtype)
        
        # Per-sequence average loss. Shape: [B]
        per_seq_loss = per_seq_kl_sum / valid_rows_per_seq

        # Finally, average across all valid sequences in the batch.
        valid_seq_mask = (valid_row_mask.sum(dim=-1) > 0)
        num_valid_seqs = valid_seq_mask.sum().clamp(min=1).to(per_seq_loss.dtype)
        
        total_loss = per_seq_loss.masked_fill(~valid_seq_mask, 0.0).sum() / num_valid_seqs
        
        return total_loss

    # Calculate the two directional KL losses using the corrected helper.
    kl_s2m = _calculate_correctly_normalized_kl(P_s2s, Q_s2m, esm_mask, esm_pair_mask)
    kl_m2s = _calculate_correctly_normalized_kl(P_m2m, Q_m2s, msa_mask, msa_pair_mask)

    # --- 4. Final Loss Aggregation ---
    total_loss = (kl_s2m + kl_m2s) / 2.0 * loss_weight

    # Final check for numerical stability.
    if not torch.isfinite(total_loss):
        logging.warning(f"residue_level_soft_label_alignment resulted in a non-finite loss: {total_loss.item()}. Returning zero.")
        return torch.zeros_like(total_loss)
        
    return total_loss


 
class DCAGraphNetwork(nn.Module):
    """
    Graph Neural Network module for Stage Three of EvoSite.
    
    Uses DCA coupling matrix as adjacency matrix for constructing a graph.
    Takes residue embeddings from Stage Two's cross-attention as node features,
    performs message passing on the graph, and outputs updated residue embeddings.

    训练策略
    根据需求建议，有两种训练此模型的方法：
        * 端到端训练：同时训练所有参数，使 GNN 能够与模型的其余部分一起适应。
        * 冻结早期阶段：冻结第一阶段和第二阶段的参数，仅训练第三阶段（GNN + 最终分类器）。这将重点关注共同进化关系的学习。
        该实现通过控制哪些参数的 require_grad=True 来支持这两种方法。
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_layers: int = 3, 
        dropout: float = 0.1, 
        threshold: float = 0.05,
        top_k: Optional[int] = None,
        gnn_type: str = "gcn",
        use_batch_norm: bool = True,
        add_residual: bool = True
    ):
        """
        Initialize the DCA Graph Network.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden node features
            output_dim: Dimension of output node features
            num_layers: Number of GNN layers
            dropout: Dropout probability
            threshold: Threshold for edge filtering based on DCA values
            top_k: If provided, only keep top-k edges per node
            gnn_type: Type of GNN layer ("sage", "gcn", "gat")
            use_batch_norm: Whether to use batch normalization
            add_residual: Whether to add residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.threshold = threshold
        self.top_k = top_k
        self.gnn_type = gnn_type.lower()
        self.use_batch_norm = use_batch_norm
        self.add_residual = add_residual
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if self.gnn_type == "sage":
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif self.gnn_type == "gcn":
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == "gat":
                self.layers.append(GATConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
            )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    


    def _create_edge_index_from_adjacency(
        self, 
        adjacency_matrix: torch.Tensor, 
        threshold: float = 0.0,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edge index and edge weights from adjacency matrix.
        
        本函数一次处理单个样本

        Args:
            adjacency_matrix: DCA coupling matrix of shape [valid_len_i, [valid_len_i]
            threshold: Threshold for edge filtering
            top_k: If provided, only keep top-k edges per node
            
        Returns:
            Tuple of (edge_index, edge_weight)
        """
        seq_len = adjacency_matrix.size(0)
        device = adjacency_matrix.device
        
        # Apply threshold to adjacency matrix
        if threshold > 0:
            mask = adjacency_matrix > threshold
            rows, cols = torch.nonzero(mask, as_tuple=True)
            weights = adjacency_matrix[rows, cols]
        else:
            # Create edges between all pairs (except self-loops)
            indices = torch.arange(seq_len, device=device)
            rows, cols = torch.meshgrid(indices, indices, indexing='ij')
            mask = rows != cols  # Remove self-loops
            rows, cols = rows[mask], cols[mask]
            weights = adjacency_matrix[rows, cols]
        
        # If top_k is specified, keep only top-k edges per node
        if top_k is not None and top_k < seq_len - 1:
            # Process each node to keep only top-k edges
            filtered_rows = []
            filtered_cols = []
            filtered_weights = []
            
            for i in range(seq_len):
                # Get edges from node i
                node_mask = rows == i
                node_cols = cols[node_mask]
                node_weights = weights[node_mask]
                
                if len(node_weights) > top_k:
                    # Keep only top-k edges
                    _, top_indices = torch.topk(node_weights, top_k)
                    filtered_rows.append(torch.full((top_k,), i, device=device))
                    filtered_cols.append(node_cols[top_indices])
                    filtered_weights.append(node_weights[top_indices])
                else:
                    # Keep all edges if less than top_k
                    filtered_rows.append(torch.full((len(node_weights),), i, device=device))
                    filtered_cols.append(node_cols)
                    filtered_weights.append(node_weights)
            
            # Combine all filtered edges
            rows = torch.cat(filtered_rows)
            cols = torch.cat(filtered_cols)
            weights = torch.cat(filtered_weights)
        
        edge_index = torch.stack([rows, cols])
        
        return edge_index, weights
    


    def forward(
        self, 
        residue_embeddings: torch.Tensor, 
        dca_matrices: List[torch.Tensor],
        valid_residue_indices: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the DCA Graph Network.

        确保residue_embeddings和dca_matrices的special token(包括cls/bos, eos, pad)去除后，再进行接下来的处理
        
        Args:
            residue_embeddings: Tensor of shape [batch_size, seq_len, embedding_dim]，包含special token
            dca_matrices: List of tensors, each of shape [valid_len_i, valid_len_i]，不含special token
            valid_residue_indices:list of tensors with indices of valid positions for each sequence
            
        Returns:
            Updated residue embeddings, shape [batch_size, seq_len, output_dim]，其中special token的位置用0填充
        """
        batch_size, max_seq_len, _ = residue_embeddings.size()
        device = residue_embeddings.device
        
        # Process each example in the batch separately
        outputs = []
        
        for b in range(batch_size):
            # Get valid indices either from provided indices or create from attention_mask
            if valid_residue_indices is not None and b < len(valid_residue_indices):
                valid_idx = valid_residue_indices[b]
            else:
                raise ValueError(
                    f"Valid residue indices not provided for batch index {b}. "
                    "Please provide valid_residue_indices or ensure attention_mask is used."
                )
            
            valid_len = len(valid_idx)
            valid_embeddings = residue_embeddings[b, valid_idx]  # [valid_len_i, embedding_dim]
            dca_matrix = dca_matrices[b]  # [valid_len_i, valid_len_i]
            
            # Ensure DCA matrix has correct shape
            if dca_matrix.size(0) != valid_len or dca_matrix.size(1) != valid_len:
                raise ValueError(f"DCA matrix shape {dca_matrix.shape} doesn't match number of valid residues {valid_len}")
            
            # Project input to hidden dimension
            x = self.input_projection(valid_embeddings)  # [valid_len_i, hidden_dim]
            
            # Create edge index and weights from DCA matrix
            edge_index, edge_weight = self._create_edge_index_from_adjacency(
                dca_matrix, self.threshold, self.top_k
            )
            
            # Store original features for residual connection
            original_x = x
            
            # Apply GNN layers
            for i, layer in enumerate(self.layers):
                # Apply graph convolution
                if self.gnn_type == "sage" or self.gnn_type == "gat":
                    x = layer(x, edge_index)
                elif self.gnn_type == "gcn":
                    x = layer(x, edge_index, edge_weight)
                
                # Apply batch normalization if enabled
                if self.use_batch_norm:
                    x = self.batch_norms[i](x)
                
                # Apply non-linearity and dropout
                x = F.relu(x)
                x = self.dropout(x)
            
            # Add residual connection if enabled
            if self.add_residual:
                x = x + original_x
            
            # Apply output projection
            x = self.output_projection(x)  # [valid_len_i, output_dim]
            
            # Create output tensor and place updated embeddings at valid positions
            padded_output = torch.zeros(max_seq_len, self.output_dim, device=device)
            padded_output[valid_idx] = x
            
            outputs.append(padded_output)
        
        # Stack all outputs
        return torch.stack(outputs)  # [batch_size, seq_len, output_dim]



class ConditionNet(nn.Module):
    def __init__(
        self,
        fusion_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        film_bottleneck: int = 64,
        synergy_hidden_dim: int = 256,
        global_max_seq_len: int = 1026,  # 最大序列长度参数
        use_residual: bool = True,
        dca_norm_method: str = "tanh"  # "sigmoid", "tanh", "none"
    ):
        """
        Initialize the Conditional Network with fixed-length processing strategy.
        
        Args:
            fusion_dim: Dimension of input fused embeddings
            output_dim: Output dimension (defaults to fusion_dim if None)
            dropout: Dropout probability
            film_bottleneck: Bottleneck dimension for FiLM MLP
            synergy_hidden_dim: Hidden dimension for synergy network MLP
            global_max_seq_len: Maximum sequence length to handle
            use_residual: Whether to use residual connections
            dca_norm_method: How to normalize DCA single-site potentials
        """
        super().__init__()
        
        if output_dim is None:
            output_dim = fusion_dim
            
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.dca_norm_method = dca_norm_method
        self.global_max_seq_len = global_max_seq_len
        
        # Step 1: Conservation score to FiLM parameters
        self.conservation_film = nn.Sequential(
            nn.Linear(1, film_bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(film_bottleneck, fusion_dim * 2)  # 输出gamma和beta
        )
        
        # Step 2: Synergy network with fixed input dimensions
        self.synergy_network = nn.Sequential(
            nn.Linear(20 + self.global_max_seq_len, synergy_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(synergy_hidden_dim, synergy_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(synergy_hidden_dim, fusion_dim),
        )
        
        # Step 3: Dynamic gate fusion
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),  # 输入: 拼接的调制特征和协同特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()  # 门控值在[0,1]范围
        )
        
        # LayerNorm + projections
        self.layer_norm = nn.LayerNorm(output_dim)
        self.output_projection = (
            nn.Linear(fusion_dim, output_dim) if fusion_dim != output_dim else nn.Identity()
        )
        if use_residual:
            self.residual_projection = (
                nn.Linear(fusion_dim, output_dim) if fusion_dim != output_dim else nn.Identity()
            )
    


    def forward(
        self,
        fused_embeddings: torch.Tensor,
        conservation_scores: torch.Tensor,
        dca_single_potentials: Union[List[torch.Tensor], torch.Tensor],
        coupling_matrix: Union[List[torch.Tensor], torch.Tensor],
        valid_residue_indices: Optional[List[torch.Tensor]] = None,
        valid_tokens_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            fused_embeddings: Fused embeddings [batch_size, seq_len, fusion_dim]
            conservation_scores: Conservation scores [batch_size, seq_len]
            dca_single_potentials: DCA single-site potentials, List of [valid_len_i, 20] tensors
            coupling_matrix: DCA coupling matrices, List of [valid_len_i, valid_len_i] tensors
            valid_residue_indices: List of Valid residue indices for each sequence in batch
            valid_tokens_mask: Valid token mask [batch_size, seq_len], 1=valid residue, 0=special token
            
        Returns:
            Conditioned embeddings [batch_size, seq_len, output_dim]
        """

        batch_size, seq_len, _ = fused_embeddings.shape
        device = fused_embeddings.device
        
        # Step 1: 使用保守性得分进行FiLM条件调制
        if conservation_scores.dim() == 2:
            conservation_scores = conservation_scores.unsqueeze(-1)  # [batch_size, seq_len, 1]
        conservation_scores = conservation_scores.to(device)

        film_params = self.conservation_film(conservation_scores)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = torch.sigmoid(gamma) * 2  # 缩放到[0, 2]范围
        regulated_embeddings = gamma * fused_embeddings + beta
        

        # Step 2: 协同特征网络处理
        synergy_features = torch.zeros_like(regulated_embeddings)
        
        for b in range(batch_size):
            valid_idx = valid_residue_indices[b] if (valid_residue_indices is not None and b < len(valid_residue_indices)) else torch.arange(seq_len, device=device)
            valid_len = valid_idx.numel()
            if valid_len == 0:
                continue
            
            # 提取特征
            potentials = dca_single_potentials[b].to(device=device, dtype=torch.float32) if isinstance(dca_single_potentials, list) else dca_single_potentials[b, valid_idx].to(device=device, dtype=torch.float32)
            matrix = coupling_matrix[b].to(device=device, dtype=torch.float32) if isinstance(coupling_matrix, list) else coupling_matrix[b, valid_idx][:, valid_idx].to(device=device, dtype=torch.float32)
            
            # 归一化单点势
            if self.dca_norm_method == "sigmoid":
                potentials = torch.sigmoid(potentials)
            elif self.dca_norm_method == "tanh":
                potentials = torch.tanh(potentials)
            
            # Start with zeros then scatter coupling values into real columns.
            aligned_coupling = torch.zeros((valid_len, self.global_max_seq_len), device=device, dtype=matrix.dtype)
            aligned_coupling[:, valid_idx] = matrix  # (valid_len × valid_len) placed into full L dimension

            # Z‑score normalisation per row to mitigate 0‑pad bias
            mean = aligned_coupling.mean(dim=1, keepdim=True)
            std = aligned_coupling.std(dim=1, keepdim=True) + 1e-8
            aligned_coupling = (aligned_coupling - mean) / std

            synergy_input = torch.cat([potentials, aligned_coupling], dim=1)          # (valid_len × (20+L))
            pos_synergy = self.synergy_network(synergy_input)   

            synergy_features[b, valid_idx] = pos_synergy


        # Step 3: 动态门控融合 - 使用显式掩码
        concat_features = torch.cat([regulated_embeddings, synergy_features], dim=-1)  # B × L × 2d
        concat_features = torch.where(
            valid_tokens_mask.unsqueeze(-1), concat_features, torch.zeros_like(concat_features)
        )
        gate = self.gate_network(concat_features)
        gate = torch.where(
            valid_tokens_mask.unsqueeze(-1), gate, torch.ones_like(gate)
        )
        final_embeddings = gate * regulated_embeddings + (1.0 - gate) * synergy_features
        
    
        
        # Step 4
        final_embeddings = self.output_projection(final_embeddings)
        final_embeddings = self.layer_norm(final_embeddings)
        if self.use_residual:
            final_embeddings = final_embeddings + self.residual_projection(fused_embeddings)

        return final_embeddings



class RelativePositionBias(nn.Module):
    """
    Learnable relative position bias module, inspired by T5 and AlphaFold2. (Vectorized & Device-Safe)
    This module adds a learnable bias to the attention logits based on the
    relative distance between query and key residues.
    """
    def __init__(self, num_heads: int, num_buckets: int, max_distance: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        # The embedding table for the relative position buckets.
        self.relative_attention_bias = nn.Embedding(self.num_buckets, num_heads)



    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int, max_distance: int) -> torch.Tensor:
        num_half_buckets = num_buckets // 2
        # 一半桶内的所有位置都直接映射
        max_exact = num_half_buckets // 2
        
        # 存储符号并使用绝对值进行操作
        sign = torch.sign(relative_position)
        abs_pos = torch.abs(relative_position)

        # 为精确映射范围内的位置创建一个布尔掩码
        is_small = abs_pos < max_exact

        # 对于较大的位置，映射到对数尺度的桶。
        # 确保用于对数计算的临时张量在正确的设备上并且是浮点数。
        # 为避免除以零，将 abs_pos 中的零替换为 1e-6
        safe_abs_pos = abs_pos.float().clone()
        safe_abs_pos[safe_abs_pos < max_exact] = max_exact
        
        log_ratio = torch.log(safe_abs_pos / max_exact) / torch.log(
            torch.tensor(max_distance / max_exact, device=abs_pos.device, dtype=torch.float)
        )
        
        val_if_large = max_exact + (log_ratio * (num_half_buckets - max_exact)).int()
        
        # 将值裁剪到最大桶索引以处理潜在的溢出
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_half_buckets - 1))

        bucket_indices = torch.where(is_small, abs_pos, val_if_large)
        
        # 重新引入符号信息以区分向前和向后的相对位置
        bucket_indices = torch.where(sign < 0, bucket_indices + num_half_buckets, bucket_indices)
        return bucket_indices



    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        为给定的序列长度计算相对位置偏置。
        """
        # 在正确的设备上创建位置索引
        q_pos = torch.arange(seq_len, dtype=torch.long, device=device)
        k_pos = torch.arange(seq_len, dtype=torch.long, device=device)
        
        # 创建一个相对位置矩阵: [seq_len, seq_len]
        relative_position = k_pos[None, :] - q_pos[:, None]
        
        # 将相对位置分桶
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        
        # 从嵌入表中查找偏置并转换为正确的 dtype 以进行混合精度训练
        bias = self.relative_attention_bias(rp_bucket).to(dtype=dtype)
        
        # 重塑以广播到注意力矩阵: [1, num_heads, seq_len, seq_len]
        return bias.permute(2, 0, 1).unsqueeze(0)



class SynergyAttentionNetwork(nn.Module):
    """
    processes the entire batch at once.
    """
    def __init__(self, fusion_dim: int, synergy_hidden_dim: int, num_heads: int, dropout: float, use_relative_pos: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = synergy_hidden_dim // num_heads
        self.use_relative_pos = use_relative_pos
        
        kv_input_dim = 1 + 20 + 1  # coupling (1) + potentials (20) + conservation (1)
        
        self.query_proj = nn.Linear(fusion_dim, synergy_hidden_dim)
        self.key_proj = nn.Linear(kv_input_dim, synergy_hidden_dim)
        self.value_proj = nn.Linear(kv_input_dim, synergy_hidden_dim)
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(synergy_hidden_dim),
            nn.Linear(synergy_hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.use_relative_pos:
            self.relative_pos_bias = RelativePositionBias(num_heads=num_heads, num_buckets=32, max_distance=128)



    def forward(
        self, 
        embeddings: torch.Tensor, 
        coupling_matrix: torch.Tensor, 
        dca_single_potentials: torch.Tensor,
        conservation_scores: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        kv_features[b, i, j, :] 这个长度为22的向量，包含了当 i 关注 j 时所需的三类关键信息：
            i 和 j 之间的直接关系（耦合强度）。
            j 的内在化学属性偏好（单点势）。
            j 的进化重要性（保守性分数）。

        Args:
            embeddings (torch.Tensor): Input embeddings. Shape: [B, L, fusion_dim].
            coupling_matrix (torch.Tensor): Pairwise coupling. Shape: [B, L, L].
            dca_single_potentials (torch.Tensor): Single-site potentials. Shape: [B, L, 20].
            conservation_scores (torch.Tensor): Conservation. Shape: [B, L, 1].
            attention_mask (torch.Tensor): Mask for padding. Shape: [B, L].

        Returns:
            torch.Tensor: Computed synergy features. Shape: [B, L, fusion_dim].
        """
        batch_size, seq_len, _ = embeddings.shape
        device, dtype = embeddings.device, embeddings.dtype

        queries = self.query_proj(embeddings)
        
        kv_features = torch.cat([
            coupling_matrix.unsqueeze(-1),
            dca_single_potentials.unsqueeze(1).expand(-1, seq_len, -1, -1), # [B, L, L, 20]
            conservation_scores.unsqueeze(1).expand(-1, seq_len, -1, -1)   # [B, L, L, 1]
        ], dim=-1)
        
        keys = self.key_proj(kv_features)
        values = self.value_proj(kv_features)
        
        q = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L, D_h]
        # 注意：这里的k和v是每个query位置i对应一套key和value，形状为[B, H, L, L, D_h]
        # k[b, h, i, j, :] 表示query i看到的key j
        k = keys.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)   # [B, L, L, H*D_h] -> [B, L, L, H, D_h] -> [B, H, L, L, D_h]
        v = values.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)   # [B, L, L, H*D_h] -> [B, L, L, H, D_h] -> [B, H, L, L, D_h]

        # 因为F.scaled_dot_product_attention不支持这种 per-query-key 的形式，
        # 需要将查询维度L视为一个批次维度来处理，进行 B*L 次独立的注意力计算。每次计算中，query是一个，key和value是完整的序列。

        # 从填充掩码创建注意力偏置
        # attn_bias形状: [B, 1, L, L]
        attn_bias = (attention_mask[:, None, None, :] * attention_mask[:, None, :, None]).float()
        fill_value = torch.finfo(dtype).min
        attn_bias = (1.0 - attn_bias) * fill_value

        if self.use_relative_pos:
            # self.relative_pos_bias(seq_len, device, dtype)返回形状: [1, H, L, L] 
            # 原attn_bias形状: [B, 1, L, L]
            # 相加后attn_bias形状: [B, H, L, L]
            attn_bias = attn_bias + self.relative_pos_bias(seq_len, device, dtype)
        else:
            # 如果不使用相对位置偏置, 需要手动扩展 attn_bias 以匹配头的数量
            # [B, 1, L, L] -> [B, H, L, L]
            attn_bias = attn_bias.expand(-1, self.num_heads, -1, -1)
            
        # 为了与`F.scaled_dot_product_attention`兼容，需要重塑注意力操作
        # 我们将查询长度维度 (L) 视为注意力的批次维度
        q_reshaped = q.transpose(1, 2).reshape(batch_size * seq_len, 1, self.num_heads, self.head_dim).transpose(1, 2) # [B*L, H, 1, D_h]
        
        # k和v的重塑，使其对每个(B*L)的query都可见
        k_reshaped = k.permute(0, 2, 3, 1, 4).reshape(batch_size * seq_len, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [B*L, H, L, D_h]
        v_reshaped = v.permute(0, 2, 3, 1, 4).reshape(batch_size * seq_len, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [B*L, H, L, D_h]
        
        # [B, H, L_q, L_k] -> [B, L_q, H, L_k]
        attn_bias_permuted = attn_bias.permute(0, 2, 1, 3)

        # [B, L_q, H, L_k] -> [B*L, H, L_k] -> [B*L, H, 1, L_k]
        attn_bias_reshaped = attn_bias_permuted.reshape(
            batch_size * seq_len, self.num_heads, seq_len
        ).unsqueeze(2)

        # attn_output 的形状将是 [B*L, H, 1, D_h]
        attn_output = F.scaled_dot_product_attention(
            q_reshaped, k_reshaped, v_reshaped, attn_mask=attn_bias_reshaped, dropout_p=0.1 if self.training else 0.0
        )
        
        # 最终 attn_output 形状: [B, L, synergy_hidden_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        synergy_features = self.output_mlp(attn_output)
        
        return synergy_features



class ConditionNet2(nn.Module):
    def __init__(
        self,
        fusion_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        synergy_hidden_dim: int = 256,
        synergy_attention_heads: int = 8,
        use_residual: bool = True,
        use_relative_pos_in_synergy: bool = False,
    ):
        super().__init__()
        
        self.output_dim = output_dim if output_dim is not None else fusion_dim
        self.fusion_dim = fusion_dim
        self.use_residual = use_residual
        
        self.synergy_network = SynergyAttentionNetwork(
            fusion_dim=fusion_dim,
            synergy_hidden_dim=synergy_hidden_dim,
            num_heads=synergy_attention_heads,
            dropout=dropout,
            use_relative_pos=use_relative_pos_in_synergy
        )
        
        self.synergy_weight = nn.Parameter(torch.tensor(0.1))
        self.pre_norm = nn.LayerNorm(self.fusion_dim)
        self.output_projection = nn.Linear(self.fusion_dim, self.output_dim)
        
        if self.use_residual and self.fusion_dim != self.output_dim:
            self.residual_projection = nn.Linear(self.fusion_dim, self.output_dim)
        else:
            self.residual_projection = nn.Identity()



    def forward(
        self,
        fused_embeddings: torch.Tensor,
        conservation_scores: torch.Tensor,
        dca_single_potentials: List[torch.Tensor], # MODIFIED: 接收一个未填充张量的列表
        coupling_matrix: List[torch.Tensor],       # MODIFIED: 接收一个未填充张量的列表
        valid_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = fused_embeddings.shape
        device = fused_embeddings.device
        dtype = fused_embeddings.dtype

        # =========================================== Top@padding ====================================================== 
        padded_potentials = torch.zeros(batch_size, seq_len, 20, device=device, dtype=dtype)
        padded_coupling = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=dtype)


        for i in range(batch_size):
            mask_i = valid_tokens_mask[i]
            valid_len = mask_i.sum().item()

            if valid_len > 0:
                if dca_single_potentials[i] is not None:
                    assert dca_single_potentials[i].shape[0] == valid_len, f"Mismatch in sample {i}: mask expects {valid_len} residues, but potentials have {dca_single_potentials[i].shape[0]}"
                    padded_potentials[i, mask_i] = dca_single_potentials[i].to(dtype=dtype)
                if coupling_matrix[i] is not None:
                    assert coupling_matrix[i].shape[0] == valid_len and coupling_matrix[i].shape[1] == valid_len, f"Mismatch in sample {i}: mask expects [{valid_len},{valid_len}], but coupling matrix is {coupling_matrix[i].shape}"
                    mask_2d = mask_i.unsqueeze(1) & mask_i.unsqueeze(0)
                    padded_coupling[i][mask_2d] = coupling_matrix[i].flatten().to(dtype=dtype)
        # =========================================== Bottom@padding ====================================================== 
        

        # =========================================== Top@normalization ====================================================== 
        potentials_norm = torch.tanh(padded_potentials)
        potentials_norm = potentials_norm * valid_tokens_mask.unsqueeze(-1)

        coupling_norm = torch.tanh(padded_coupling)
        coupling_norm = coupling_norm * (valid_tokens_mask.unsqueeze(-1) * valid_tokens_mask.unsqueeze(-2))

        normalized_fused_embeddings = self.pre_norm(fused_embeddings)
        # =========================================== Bottom@normalization ======================================================


        synergy_features = self.synergy_network(
            embeddings=normalized_fused_embeddings,
            coupling_matrix=coupling_norm,
            dca_single_potentials=potentials_norm,
            conservation_scores=conservation_scores.unsqueeze(-1),
            attention_mask=valid_tokens_mask,
        )
        
        enhanced_embeddings = normalized_fused_embeddings + self.synergy_weight * synergy_features
        projected_output = self.output_projection(enhanced_embeddings)

        # Apply the main residual connection
        if self.use_residual:
            # Add the original, un-normalized fused_embeddings
            final_embeddings = self.residual_projection(normalized_fused_embeddings) + projected_output
        else:
            final_embeddings = projected_output
            
        # Final mask to ensure padding is zero
        final_embeddings = final_embeddings * valid_tokens_mask.unsqueeze(-1)
        
        return final_embeddings



class ConditionNet3(nn.Module):
    def __init__(
        self,
        fusion_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        synergy_hidden_dim: int = 256,
        synergy_attention_heads: int = 8,
        use_residual: bool = True,
        use_relative_pos_in_synergy: bool = False,
        use_coupling_norm: bool = False,
        use_coupling_binary: bool = False,
        coupling_thresh: float = 0.0,
    ):
        """
        Args:
            use_coupling_norm (bool): If True, applies min-max normalization to the coupling matrix.
            use_coupling_binary (bool): If True, binarizes the coupling matrix (non-zero values become 1).
            coupling_thresh (float): Threshold to filter noise from the coupling matrix. Values below are set to 0.
        """
        super().__init__()
        
        self.output_dim = output_dim if output_dim is not None else fusion_dim
        self.fusion_dim = fusion_dim
        self.use_residual = use_residual
        self.use_coupling_norm = use_coupling_norm
        self.use_coupling_binary = use_coupling_binary
        self.coupling_thresh = coupling_thresh
        
        self.synergy_network = SynergyAttentionNetwork(
            fusion_dim=fusion_dim,
            synergy_hidden_dim=synergy_hidden_dim,
            num_heads=synergy_attention_heads,
            dropout=dropout,
            use_relative_pos=use_relative_pos_in_synergy
        )
        
        self.synergy_weight = nn.Parameter(torch.tensor(0.1))
        self.pre_norm = nn.LayerNorm(self.fusion_dim)
        self.output_projection = nn.Linear(self.fusion_dim, self.output_dim)
        
        if self.use_residual and self.fusion_dim != self.output_dim:
            self.residual_projection = nn.Linear(self.fusion_dim, self.output_dim)
        else:
            self.residual_projection = nn.Identity()



    def forward(
        self,
        fused_embeddings: torch.Tensor,
        conservation_scores: torch.Tensor,
        dca_single_potentials: List[torch.Tensor],
        coupling_matrix: List[torch.Tensor],
        valid_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = fused_embeddings.shape
        device = fused_embeddings.device
        dtype = fused_embeddings.dtype

        padded_potentials = torch.zeros(batch_size, seq_len, 20, device=device, dtype=dtype)
        padded_coupling = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=dtype)

        for i in range(batch_size):
            mask_i = valid_tokens_mask[i]
            valid_len = mask_i.sum().item()

            if valid_len > 0:
                if dca_single_potentials[i] is not None:
                    assert dca_single_potentials[i].shape[0] == valid_len, f"Mismatch in sample {i}: mask expects {valid_len} residues, but potentials have {dca_single_potentials[i].shape[0]}"
                    padded_potentials[i, mask_i] = dca_single_potentials[i].to(dtype=dtype)
                if coupling_matrix[i] is not None:
                    assert coupling_matrix[i].shape[0] == valid_len and coupling_matrix[i].shape[1] == valid_len, f"Mismatch in sample {i}: mask expects [{valid_len},{valid_len}], but coupling matrix is {coupling_matrix[i].shape}"
                    current_coupling = coupling_matrix[i]

                    if self.use_coupling_norm:
                        eps = 1e-8
                        min_val = torch.min(current_coupling)
                        max_val = torch.max(current_coupling)
                        denominator = max_val - min_val
                        if denominator > eps:
                            current_coupling = (current_coupling - min_val) / denominator
                        else:
                            current_coupling = torch.zeros_like(current_coupling) # Or handle as a special case

                    if self.coupling_thresh > 0.0:
                        current_coupling = torch.where(
                            current_coupling > self.coupling_thresh, current_coupling, 0.0
                        )

                    if self.use_coupling_binary:
                        current_coupling = (current_coupling > 0).to(dtype=dtype)
                    
                    mask_2d = mask_i.unsqueeze(1) & mask_i.unsqueeze(0)
                    padded_coupling[i][mask_2d] = current_coupling.flatten().to(dtype=dtype)

        potentials_norm = torch.tanh(padded_potentials)
        potentials_norm = potentials_norm * valid_tokens_mask.unsqueeze(-1)

        normalized_fused_embeddings = self.pre_norm(fused_embeddings)

        synergy_features = self.synergy_network(
            embeddings=normalized_fused_embeddings,
            coupling_matrix=padded_coupling,
            dca_single_potentials=potentials_norm,
            conservation_scores=conservation_scores.unsqueeze(-1),
            attention_mask=valid_tokens_mask,
        )
        
        enhanced_embeddings = normalized_fused_embeddings + self.synergy_weight * synergy_features
        projected_output = self.output_projection(enhanced_embeddings)

        if self.use_residual:
            final_embeddings = self.residual_projection(fused_embeddings) + projected_output # Use original fused_embeddings for residual
        else:
            final_embeddings = projected_output
            
        final_embeddings = final_embeddings * valid_tokens_mask.unsqueeze(-1)
        
        return final_embeddings
    


class CouplingFeatureEncoder(nn.Module):

    def __init__(
        self,
        output_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling_method: str = 'cls',
        chunk_size: int = 1024  # New parameter to control memory-speed trade-off
    ):
        """
        Args:
            output_dim (int): The final dimension of the output evolution features.
            embed_dim (int): The internal embedding dimension.
            num_heads (int): The number of attention heads in the profile Transformer.
            dropout (float): Dropout rate used in the Transformer encoder.
            pooling_method (str): The method to aggregate profile features.
                                  Options: 'cls', 'mean', 'max'.
            chunk_size (int): The number of residue profiles to process in a single forward pass
                              through the transformer, reducing peak memory usage.
        """
        super().__init__()
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.pooling_method = pooling_method.lower()
        self.chunk_size = chunk_size
        
        if self.pooling_method not in ['cls', 'mean', 'max']:
            raise ValueError(f"Unsupported pooling_method: {pooling_method}. Choose from 'cls', 'mean', 'max'.")

        if self.pooling_method == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.feature_projection = nn.Linear(21, embed_dim) # 1 (coupling) + 20 (potentials)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.output_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )


    def forward(
        self,
        dca_single_potentials: List[torch.Tensor],
        coupling_matrix: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        potentials_expanded: [L_max, 20] -> [L_max, L_max, 20]
            | 1 | 2 | ... | L_max |
                    >>
            | 1 | 2 | ... | L_max |
            |---|---|-----|-------|
            | 1 | 2 | ... | L_max |
            | . | . | ... | ..... |
            | 1 | 2 | ... | L_max |
        
        transformer_input: [L, L, embed_dim] -> [L, L+1, embed_dim]
            | 1 | 2 | ... | L |
            |---|---|-----|---|
            | 1 | 2 | ... | L |
            | . | . | ... | . |
            | 1 | 2 | ... | L |
                    >>
            | cls | 1 | 2 | ... | L |
            |-----|---|---|-----|---|
            | cls | 1 | 2 | ... | L |
            | cls | . | . | ... | . |
            | cls | 1 | 2 | ... | L |
        Args:
            dca_single_potentials (List[torch.Tensor]): List of B unpadded potential tensors.
            coupling_matrix (List[torch.Tensor]): List of B unpadded coupling matrices.

        Returns:
            torch.Tensor: A packed/sparse tensor of synergy features. Shape: [|V_res|, output_dim].
        """
        lengths = [c.shape[0] for c in coupling_matrix]
        if not lengths:
            return torch.empty(0, self.output_dim) # Handle empty input gracefully
            
        max_len = max(lengths)
        batch_size = len(coupling_matrix)
        device = coupling_matrix[0].device

        # Step 1: Pad all tensors to the max length in the batch. This is memory-efficient.
        padded_potentials = pad_sequence([p.to(device) for p in dca_single_potentials], batch_first=True, padding_value=0.0)
        
        # Manual padding for the 2D coupling matrices
        manually_padded_couplings = []
        for c in coupling_matrix:
            l = c.shape[0]
            padded_c = F.pad(c, (0, max_len - l, 0, max_len - l), "constant", 0.0)
            manually_padded_couplings.append(padded_c)
        padded_couplings = torch.stack(manually_padded_couplings, dim=0)

        # A mask indicating valid (non-padded) residues for each sequence in the batch.
        mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]

        # Step 2: Process profiles in chunks to avoid large intermediate tensors.
        total_profiles = batch_size * max_len
        all_output_vectors = []

        for i in range(0, total_profiles, self.chunk_size):
            end = min(i + self.chunk_size, total_profiles)
            current_chunk_size = end - i
            
            # Map flat chunk indices back to (batch, residue) indices
            batch_indices = torch.arange(i, end, device=device) // max_len
            residue_indices = torch.arange(i, end, device=device) % max_len

            # Efficiently construct the input for the current chunk without large expansions.
            # Shape: [current_chunk_size, max_len]
            chunk_couplings = padded_couplings[batch_indices, residue_indices, :]
            # Shape: [current_chunk_size, max_len, 20]
            chunk_potentials = padded_potentials[batch_indices]
            
            # Combine features for the chunk. Shape: [current_chunk_size, max_len, 21]
            chunk_profile_features = torch.cat([chunk_couplings.unsqueeze(-1), chunk_potentials], dim=-1)
            # Project features to the embedding dimension. Shape: [current_chunk_size, max_len, embed_dim]
            chunk_projected_profile = self.feature_projection(chunk_profile_features)
            
            # Prepare input and mask for the Transformer.
            if self.pooling_method == 'cls':
                cls_expanded = self.cls_token.expand(current_chunk_size, -1, -1)
                transformer_input = torch.cat([cls_expanded, chunk_projected_profile], dim=1)
                
                # Create mask for the transformer: True for valid positions, False for padding.
                # CLS token is always valid.
                chunk_mask = mask[batch_indices]
                transformer_mask = F.pad(chunk_mask, (1, 0), value=1)
            else: # 'mean' or 'max' pooling
                transformer_input = chunk_projected_profile
                transformer_mask = mask[batch_indices]
            
            # The transformer expects `src_key_padding_mask` where True indicates a position to be IGNORED.
            encoded_profile = self.transformer_encoder(
                transformer_input, 
                src_key_padding_mask=~transformer_mask
            )

            # Pool the transformer output to get one vector per profile.
            if self.pooling_method == 'cls':
                vectors = encoded_profile[:, 0, :]
            else: # 'mean' or 'max' pooling
                # Mask out padding before pooling
                encoded_profile.masked_fill_(~transformer_mask.unsqueeze(-1), 0.0)
                if self.pooling_method == 'mean':
                    sum_vectors = encoded_profile.sum(dim=1)
                    num_valid_tokens = transformer_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    vectors = sum_vectors / num_valid_tokens
                else: # 'max'
                    # To do max-pooling correctly, set padding to a very small number
                    encoded_profile.masked_fill_(~transformer_mask.unsqueeze(-1), -torch.finfo(encoded_profile.dtype).max)
                    vectors, _ = encoded_profile.max(dim=1)

            all_output_vectors.append(vectors)
        
        # Step 3: Combine results from all chunks.
        # Shape: [B * L_max, embed_dim]
        combined_vectors = torch.cat(all_output_vectors, dim=0)
        # Reshape back to the batch structure. Shape: [B, L_max, embed_dim]
        packed_features = combined_vectors.view(batch_size, max_len, self.embed_dim)
        
        # Filter out features for padded residues to get a packed tensor.
        # Shape: [|V_res|, embed_dim], where |V_res| is the total number of valid residues in the batch.
        final_packed_features = packed_features[mask]

        # Final projection to the desired output dimension.
        projected_synergy_features = self.output_projection(final_packed_features)
        
        return projected_synergy_features
    
    

class CoevolutionPredictionHead(nn.Module):
    """
    Predicts pairwise co-evolutionary scores from residue pair embeddings.
    This head takes concatenated features of two residues and predicts their coupling score, intended to be trained against a DCA matrix.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        """
        Initializes the CoevolutionPredictionHead.

        Args:
            input_dim (int): The dimension of the input features for a single residue.
                             The MLP will take input_dim * 2.
            hidden_dim (int): The hidden dimension of the MLP. 256 is chosen for its balance of
                              expressive power for high-dimensional input and regularization.
            dropout (float): Dropout rate. 0.2 provides slightly stronger regularization
                             suited for the increased hidden dimension.
        """
        super().__init__()
        # A simple MLP to process the concatenated pair features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Output a single scalar value for the coupling score
        )



    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for co-evolution prediction.

        Args:
            pair_features (torch.Tensor): Concatenated features of all residue pairs in a protein.
                                          Shape: (|V_res|, |V_res|, input_dim * 2)

        Returns:
            torch.Tensor: Predicted coupling scores for each pair. Shape: (|V_res|, |V_res|)
        """
        # Squeeze the last dimension to get the final score matrix
        return self.mlp(pair_features).squeeze(-1)



class DistancePredictionHead(nn.Module):
    """
    Predicts binned pairwise residue distances from residue pair embeddings.
    This head is designed for future fine-tuning with real PDB structures. It outputs
    logits over a set of distance bins.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_bins: int = 64, dropout: float = 0.2):
        """
        Initializes the DistancePredictionHead.

        Args:
            input_dim (int): The dimension of the input features for a single residue.
                             The MLP will take input_dim * 2.
            hidden_dim (int): The hidden dimension of the MLP. 256 is recommended for consistency.
            num_bins (int): The number of distance bins to predict. 64 is chosen to align with
                            state-of-the-art structure prediction models like AlphaFold2, providing
                            a high-resolution structural signal that is learnable with a large dataset.
            dropout (float): Dropout rate. 0.2 for stronger regularization.
        """
        super().__init__()
        self.num_bins = num_bins
        # MLP to predict logits for each distance bin
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins)
        )



    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distance prediction.

        Args:
            pair_features (torch.Tensor): Concatenated features of all residue pairs in a protein.
                                          Shape: (|V_res|, |V_res|, input_dim * 2)

        Returns:
            torch.Tensor: Logits for each distance bin for each pair. Shape: (|V_res|, |V_res|, num_bins)
        """
        return self.mlp(pair_features)
    


class ConservationWeightedMlmLoss(nn.Module):
    """
    Computes a Masked Language Modeling (MLM) loss where each token's contribution
    is weighted by its evolutionary conservation score. This forces the model to
    pay more attention to highly conserved, functionally important residues.

    

    ### 标准MLM的根本局限性

    标准MLM的损失函数是**位置不可知 (Position-Agnostic)** 和 **类别不可知 (Class-Agnostic)** 的。
    *   **位置不可知**：它随机在任何位置进行mask，无论该位置是高度保守的功能核心，还是无关紧要的表面loop。
    *   **类别不可知**：它只关心能否准确预测出20种氨基酸中的那一个，而不关心这个氨基酸的化学本质（例如，把一个带负电的`D`错判成另一个带负电的`E`，其惩罚与错判成一个疏水的`V`是一样的）。

    为了凸显活性位点预测，我们的损失函数设计必须打破这两个“不可知”。

    ### 方案一：基于保守性的加权损失 (Conservation-Weighted Loss)

    这是最直接、最有效的改进，直接解决了“位置不可知”的问题。

    *   **生物化学先验**：活性位点是蛋白质中进化选择压力最强的区域，因此表现出**最高的序列保守性**。模型在这些位置犯的错误，应该比在高度可变位置犯的错误代价更高。

    *   **方法设计**：
        1.  **预计算保守性得分**：在开始训练前，对您的MSA进行一次预处理。计算每一列（每一个位置 `i`）的保守性得分 `C(i)`。计算方法可以从简单的香农熵，到更复杂的考虑物化性质的打分（如 `Scorecons`）。将得分归一化到 `[0, 1]` 区间（或一个合适的范围）。
        2.  **设计加权损失函数**：在训练过程中，对于每一个被mask的位置 `i`，计算其标准的交叉熵损失 `L_CE(i)`。最终的损失 `L(i)` 是两者的乘积，并引入一个超参数 `α` 来调节权重的影响。
            
            `L_final = Σ [ (1 + α * C(i)) * L_CE(i) ]`
            
            (对所有被mask的位置 `i` 求和)
            
            这里的 `(1 + α * C(i))` 就是权重。当一个位置 `i` 的保守性 `C(i)` 很高时，它的损失会被放大 `(1 + α)` 倍；而当保守性很低时，权重接近1，基本不受影响。

    *   **为何有效**：这种设计直接告诉模型：“**听着，在这些高度保守的位置上，你绝对不能犯错，否则将受到严厉惩罚！**” 为了最小化总体损失，模型将被迫分配更多的模型容量（attention heads, network depth）去理解和建模这些关键位札的上下文，其学习到的向量表征（embedding）会自然地富含“功能重要性”的信息。

    *   **学术思想**：这个思想来源于**代价敏感学习 (Cost-Sensitive Learning)** 和 **重要性采样 (Importance Sampling)**。虽然没有一篇专门针对蛋白质的“Conservation-Weighted Loss”的开创性论文，但这是将成熟的机器学习理论应用于该领域的直接体现。类似的思想在**主动学习（Active Learning）**中挑选信息量最大的样本时也有广泛应用。
    """
    def __init__(self, base_loss_fn: nn.Module, alpha: float = 1.0):
        """
        Args:
            base_loss_fn (nn.Module): The underlying loss function, typically CrossEntropyLoss with reduction='none'.
            alpha (float): A hyperparameter to scale the influence of conservation scores.
                           A higher alpha means more penalty for errors on conserved sites.
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, conservation_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate the conservation-weighted MLM loss.

        Args:
            logits (torch.Tensor): The model's predictions for masked tokens. Shape: (num_masked_tokens, vocab_size).
            targets (torch.Tensor): The ground truth token IDs. Shape: (num_masked_tokens,).
            conservation_scores (torch.Tensor): The conservation scores for the masked positions. Shape: (num_masked_tokens,).

        Returns:
            torch.Tensor: A scalar tensor representing the final weighted loss.
        """
        # Calculate the raw, un-reduced loss for each masked token
        per_token_loss = self.base_loss_fn(logits, targets)

        # Create the weights: (1 + alpha * conservation_score)
        # Ensure conservation_scores are on the correct device and of the correct type
        weights = 1 + self.alpha * conservation_scores.to(device=per_token_loss.device, dtype=per_token_loss.dtype)

        # Apply the weights to the per-token loss
        weighted_loss = per_token_loss * weights

        # Return the mean of the weighted losses
        return weighted_loss.mean()



class CoevolutionaryReconstructionLoss(nn.Module):
    """
    An advanced auxiliary task module that encapsulates the entire logic for
    Co-evolutionary Span Masking and Reconstruction.
    ... (docstring remains the same) ...


    
    ### 方案二：基于协同进化的结构化掩码与重建损失 (Co-evolutionary Span Masking)

    此方案最为精巧，它将ECNet的核心洞见——协同进化——直接融入到MLM的训练范式中，以模拟活性位点在空间上的协同性。

    *   **生物化学先验**：活性位点不是单个残基，而是一个在三维空间中聚集的**功能模块**。这个模块内的残基，即使在序列上相距很远，也表现出强烈的**协同进化 (Co-evolution)** 关系。标准MLM一次只mask一个残基，无法有效学习这种模块化的协同性。

    *   **方法设计**：
        1.  **预计算协同进化图谱**：使用CCMpred或DCA等工具，从MSA中计算出残基间的协同进化耦合强度矩阵。这构成了一个图（Graph），节点是残基，边的权重是耦合强度。
        2.  **设计结构化掩码策略**：在mask时，不再是随机独立地选择残基。
            *   首先，随机选择一个起始残基 `i` 进行mask。
            *   然后，基于协同进化图谱，以一定概率（或确定性地）**同时mask掉与 `i` 耦合最强的 `k` 个伙伴残基** (`j1, j2, ...`)。这相当于在协同进化网络上进行了一次“邻域采样与掩码”。
        3.  **损失函数**：损失函数仍然是所有被mask位置的重建损失之和。
            
            `L_final = L_CE(i) + L_CE(j1) + L_CE(j2) + ...`

    *   **为何有效**：这种策略的颠覆性在于，它极大地增加了模型的学习难度和目标。为了重建残基 `i`，模型无法再“偷看”它最亲密的伙伴 `j1`（因为它也被mask了）。模型必须从一个更广阔、更全局的上下文中去理解整个功能模块的作用，才能同时准确地重建出这个模块内的所有成员。这迫使模型去学习**超越局部序列窗口的、与三维结构和功能模块相关的长程依赖关系**。这正是识别活性位点网络所需的关键能力。

    *   **学术思想**：源自NLP领域的**SpanBERT (Joshi et al., 2020)** 和 **ERNIE (Baidu)**。它们通过mask连续的文本片段（span）来学习更高层次的语义概念。我们的方案则是将这个思想从“序列上的连续”推广到了“**协同进化网络上的连续**”，使其完美适配蛋白质这一物理实体。
    """
    def __init__(self, base_loss_fn: nn.Module, k: int = 3, mask_token_id: int = 32):
        """
        ... (init remains the same) ...
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.k = k
        self.mask_token_id = mask_token_id

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a fully vectorized co-evolutionary span masking on a batch of data.
        This method should be called *before* the main model's forward pass.

        Args:
            batch (Dict[str, Any]): The original batch from the data loader.
                                   Must contain 'sequences' (padded tokens), 'coupling_matrices' (list),
                                   and 'esm_valid_tokens_mask'.

        Returns:
            Dict[str, Any]: A modified batch dictionary with masked tokens and new targets for this task.
        """
        # --- 1. Get padded tensors and masks from the batch ---
        # Using .clone() to avoid modifying the original batch tensors in-place
        input_tokens = batch['sequences'].clone()
        # The valid_mask from the dataloader should be used to know the true length of each sequence
        valid_mask = batch['esm_valid_tokens_mask']
        coupling_matrices_list = batch['coupling_matrices'] # List of [L_valid, L_valid] tensors
        
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        # --- 2. Vectorized Anchor Selection ---
        # Get the true length of each sequence from the mask
        seq_lengths = valid_mask.sum(dim=1)
        # Generate a random anchor index for each sequence within its valid length
        # Using torch.rand and scaling is safer than multinomial for variable lengths
        anchor_indices = (torch.rand(batch_size, device=device) * seq_lengths.float()).long()
        
        # --- 3. Prepare Padded DCA Matrix and Find Top-K Partners ---
        padded_dca = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
        for i in range(batch_size):
            valid_len = seq_lengths[i].item()
            if valid_len > 0:
                # Place the valid DCA matrix into the padded tensor, only considering valid residue indices
                # Note: valid_mask includes BOS/EOS, but DCA matrix is for residues only.
                # Assuming valid_mask is correctly aligned with sequence content for simplicity.
                # In a real implementation, indices might need adjustment for special tokens.
                padded_dca[i, :valid_len, :valid_len] = coupling_matrices_list[i]

        # Set diagonal to a very low value to prevent self-selection
        padded_dca.diagonal(dim1=-2, dim2=-1).fill_(-torch.inf)

        # Use `torch.gather` to select the coupling scores for all anchors in the batch at once
        # anchor_indices need to be shaped correctly for gathering
        anchor_couplings = torch.gather(padded_dca, 1, anchor_indices.view(batch_size, 1, 1).expand(-1, 1, seq_len)).squeeze(1)
        
        # Mask out padding positions from being selected as partners
        anchor_couplings.masked_fill_(~valid_mask, -torch.inf)
        
        # Get the top-k partner indices for every sequence in the batch
        # `k` must be clamped to be less than the sequence length
        k_clamped = min(self.k, seq_len - 1)
        _, partner_indices = torch.topk(anchor_couplings, k=k_clamped, dim=1)

        # --- 4. Create the final mask for all tokens to be changed ---
        # Create a full batch mask initialized to False
        coevo_mask = torch.zeros_like(input_tokens, dtype=torch.bool)
        
        # Mark the anchor positions
        coevo_mask[torch.arange(batch_size), anchor_indices] = True
        
        # Mark the partner positions. `scatter_` is an efficient in-place operation.
        # We need to expand the mask for scatter.
        coevo_mask.scatter_(1, partner_indices, True)

        # --- 5. Prepare Targets and Mask the Input Tokens ---
        # Store the original token IDs that will be masked
        coevo_targets = input_tokens[coevo_mask]
        
        # Update the input tokens by placing the mask token ID at the masked positions
        input_tokens[coevo_mask] = self.mask_token_id

        # --- 6. Update the batch dictionary ---
        prepared_batch = batch.copy()
        prepared_batch['coevo_masked_input_tokens'] = input_tokens
        # The indices for the loss function are now simply the boolean mask itself
        prepared_batch['coevo_mask'] = coevo_mask
        prepared_batch['coevo_targets'] = coevo_targets
        
        return prepared_batch

    def forward(self, model_outputs: Dict[str, torch.Tensor], prepared_batch: Dict[str, Any]) -> torch.Tensor:
        """
        Calculates the reconstruction loss. This part remains mostly the same,
        but now it uses the boolean mask to select the correct logits.
        """
        if 'coevo_mask' not in prepared_batch or prepared_batch['coevo_mask'].sum() == 0:
            return torch.tensor(0.0, device=model_outputs['logits'].device)

        if 'lm_head_logits' not in model_outputs:
             raise KeyError("Model output must contain 'lm_head_logits' for co-evolutionary reconstruction loss.")

        all_logits = model_outputs['lm_head_logits'] # Shape: (batch_size, seq_len, vocab_size)
        
        coevo_mask = prepared_batch['coevo_mask']     # Shape: (batch_size, seq_len)
        targets = prepared_batch['coevo_targets']     # Shape: (num_masked_tokens,)

        # Select the logits for the positions that were masked using the boolean mask
        coevo_logits = all_logits[coevo_mask] # Shape: (num_masked_tokens, vocab_size)
        
        return self.base_loss_fn(coevo_logits, targets)



class DynamicAttentionPooling(nn.Module):
    """
    根据查询向量（如底物表征），使用注意力机制对一组候选向量（如活性位点残基）进行加权池化。
    (修正版，正确实现了底物条件化注意力)
    """
    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim (int): 输入的蛋白质残基和底物表征的维度。
        """
        super().__init__()
        # 注意力网络现在处理蛋白质和底物特征的组合
        # 输入维度是 feature_dim，因为我们将使用加法或拼接的方式组合特征
        self.attention_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, 
                query_vector: torch.Tensor, 
                candidate_vectors: torch.Tensor, 
                candidate_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_vector (torch.Tensor): 查询向量 (底物表征), shape (B, D).
            candidate_vectors (torch.Tensor): 候选向量集合 (蛋白质残基特征), shape (B, L_cand, D).
            candidate_mask (torch.Tensor): 候选向量的掩码, shape (B, L_cand), True表示有效.

        Returns:
            torch.Tensor: 动态池化后的向量, shape (B, D).
        """
        B, L_cand, D = candidate_vectors.shape
        
        # --- 核心修正 ---
        # 1. 将查询向量 (底物) 扩展，使其能与每个候选残基进行交互
        query_expanded = query_vector.unsqueeze(1).expand(-1, L_cand, -1) # (B, L_cand, D)

        # 2. 组合底物和残基特征。加法是一种高效且有效的交互方式。
        #    这使得每个残基的注意力分数都依赖于底物。
        combined_features = query_expanded + candidate_vectors 
        
        # 3. 通过注意力网络计算对齐分数
        attention_logits = self.attention_network(combined_features).squeeze(-1) # (B, L_cand)
        # --- 修正结束 ---

        # 应用掩码，将无效位置的 logits 设置为一个很大的负数
        attention_logits.masked_fill_(~candidate_mask, -1e9)

        # Softmax 得到权重
        attention_weights = F.softmax(attention_logits, dim=1) # (B, L_cand)

        # 加权求和，构建动态口袋表征
        dynamic_representation = torch.sum(attention_weights.unsqueeze(-1) * candidate_vectors, dim=1) # (B, D)
        
        return dynamic_representation



class AdaptiveContrastiveLoss(nn.Module):
    """
    自适应对比学习损失模块（完整修正版）。
    """
    def __init__(self, feature_dim: int, projection_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

        self.protein_projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, projection_dim))
        self.substrate_projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, projection_dim))
        
        # 将动态池化模块实例化
        self.attention_pooling = DynamicAttentionPooling(projection_dim)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, 
                protein_features: torch.Tensor, 
                substrate_features: torch.Tensor, 
                active_site_mask: torch.Tensor,
                protein_valid_mask: torch.Tensor,
                substrate_valid_mask: torch.Tensor) -> torch.Tensor:
        
        # 1. 投影特征并归一化
        projected_protein = F.normalize(
            self.protein_projection_head(protein_features * protein_valid_mask.unsqueeze(-1)), 
            p=2, dim=-1
        )
        projected_substrate = F.normalize(
            self.substrate_projection_head(substrate_features * substrate_valid_mask.unsqueeze(-1)), 
            p=2, dim=-1
        )
        
        # 2. 池化底物表征
        substrate_lengths = substrate_valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        rep_substrate = torch.sum(projected_substrate * substrate_valid_mask.unsqueeze(-1), dim=1) / substrate_lengths
        
        # 3. 动态构建活性口袋表征
        valid_candidates_mask = active_site_mask & protein_valid_mask
        rep_dynamic_pocket = self.attention_pooling(
            query_vector=rep_substrate, 
            candidate_vectors=projected_protein,
            candidate_mask=valid_candidates_mask
        )

        # 4. 计算 InfoNCE 损失
        sim_matrix = torch.matmul(rep_substrate, rep_dynamic_pocket.T) / self.temperature
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
        loss = self.loss_fn(sim_matrix, labels)
        
        return loss



class PromptAttentionPredictor(nn.Module):
    """
    Prompt-based active site prediction module inspired by semantic segmentation.
    
    This module implements learnable prompt vectors for each site type, computing
    similarity scores between residue embeddings and prompts for classification.
    类似于语义分割中每个类别有一个learnable prototype，我们为每种site-type定义一个learnable prompt向量。
    
    Site Types in EvoSite:
    - 0: Background (non-active site)
    - 1: Binding site
    - 2: Catalytic site  
    - 3: Other site
    """
    
    def __init__(
        self,
        input_dim: int,
        num_site_types: int = 4,
        prompt_init_method: str = "random",  # "random", "physico_chemical", "normal"
        temperature: float = 1.0,
        learnable_temperature: bool = True,
        use_multi_head_attention: bool = False,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        use_layer_norm: bool = True,
        prompt_regularization: float = 0.01,
        use_prompt_diversity_loss: bool = True,
        diversity_loss_weight: float = 0.1,
        similarity_metric: str = "dot_product",  # "dot_product", "cosine", "scaled_dot_product"
        use_residual_connection: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize prompt-based predictor with enhanced features.
        """
        super(PromptAttentionPredictor, self).__init__()
        
        # Save prompt parameters
        self.input_dim = input_dim
        self.num_site_types = num_site_types
        self.prompt_init_method = prompt_init_method
        self.learnable_temperature = learnable_temperature
        self.use_multi_head_attention = use_multi_head_attention
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.use_layer_norm = use_layer_norm
        self.prompt_regularization = prompt_regularization
        self.use_prompt_diversity_loss = use_prompt_diversity_loss
        self.diversity_loss_weight = diversity_loss_weight
        self.similarity_metric = similarity_metric
        self.use_residual_connection = use_residual_connection
        self.dropout = dropout
        
        # Set default values for removed parameters to maintain functionality
        self.prompt_l2_norm = True  # Default: apply L2 normalization for stability
        self.diversity_loss_type = "cosine"  # Default: use cosine similarity for diversity
        self.eps = 1e-8  # Default: numerical stability parameter
        self.temperature_min = 0.1  # Default: minimum temperature boundary
        self.temperature_max = 10.0  # Default: maximum temperature boundary
        
        # Initialize learnable prompt vectors for each site type
        self.prompts = nn.Parameter(torch.randn(num_site_types, input_dim))
        self._initialize_prompts()
        
        # Temperature parameter for scaling similarities with bounds
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # Optional multi-head attention for more sophisticated prompt matching
        if use_multi_head_attention:
            self.prompt_attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_attention_heads,
                dropout=attention_dropout,
                batch_first=True
            )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
        else:
            self.layer_norm = None
            
        # Residual projection if needed
        if use_residual_connection:
            self.residual_proj = nn.Linear(input_dim, num_site_types)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Site type names for interpretability
        self.site_type_names = [
            "Background", "Binding", "Catalytic", "Other"
        ][:num_site_types]
        
        logging.info(f"Initialized PromptAttentionPredictor with {num_site_types} prompt vectors")
        logging.info(f"Prompt initialization method: {prompt_init_method}")
        logging.info(f"Similarity metric: {similarity_metric}")
    


    def _initialize_prompts(self):
        """Initialize prompt vectors with improved methods."""
        if self.prompt_init_method == "random":
            # Improved Xavier uniform with proper scaling
            bound = math.sqrt(6.0 / (self.input_dim + self.num_site_types))
            nn.init.uniform_(self.prompts, -bound, bound)
            
        elif self.prompt_init_method == "normal":
            # Improved normal initialization with better scaling
            std = math.sqrt(2.0 / self.input_dim)
            nn.init.normal_(self.prompts, mean=0.0, std=std)
            
        elif self.prompt_init_method == "physico_chemical":
            with torch.no_grad():
                # Background: neutral/zero vector with small noise
                self.prompts[0].normal_(0.0, 0.01)
                
                # Binding site: positive values for hydrophobic/aromatic properties
                self.prompts[1].normal_(0.2, 0.05).clamp_(0.0, 1.0)
                
                # Catalytic site: mixed positive/negative for charged/polar properties  
                self.prompts[2].normal_(0.0, 0.1)
                
                # Other site: orthogonal to other types
                if self.num_site_types > 3:
                    self.prompts[3].normal_(0.0, 0.05)
                    # Make it orthogonal to binding and catalytic
                    for i in range(1, 3):
                        dot_product = torch.dot(self.prompts[3], self.prompts[i])
                        self.prompts[3] -= dot_product * self.prompts[i] / torch.norm(self.prompts[i])**2
        
        elif self.prompt_init_method == "orthogonal":
            # 新增：正交初始化方法
            nn.init.orthogonal_(self.prompts)
            
        else:
            raise ValueError(f"Unknown prompt initialization method: {self.prompt_init_method}")
        
        # Apply L2 normalization if enabled
        if self.prompt_l2_norm:
            with torch.no_grad():
                self.prompts.data = F.normalize(self.prompts.data, dim=-1)
    


    def _get_clamped_temperature(self):
        """Get temperature clamped to valid range."""
        return torch.clamp(self.temperature, self.temperature_min, self.temperature_max)
    


    def compute_prompt_similarity(
        self, 
        residue_embeddings: torch.Tensor,
        prompts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity with enhanced numerical stability.
        """
        if self.similarity_metric == "dot_product":
            similarity = torch.einsum('bld,kd->blk', residue_embeddings, prompts)
            
        elif self.similarity_metric == "scaled_dot_product":
            similarity = torch.einsum('bld,kd->blk', residue_embeddings, prompts)
            similarity = similarity / math.sqrt(self.input_dim)
            
        elif self.similarity_metric == "cosine":
            # Enhanced numerical stability
            residue_norm = torch.norm(residue_embeddings, dim=-1, keepdim=True)
            prompt_norm = torch.norm(prompts, dim=-1, keepdim=True)
            
            # Add epsilon for numerical stability
            residue_normalized = residue_embeddings / (residue_norm + self.eps)
            prompt_normalized = prompts / (prompt_norm + self.eps)
            
            similarity = torch.einsum('bld,kd->blk', residue_normalized, prompt_normalized)
            
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity
    


    def forward(
        self, 
        residue_embeddings: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through prompt attention predictor with enhanced features.
        
        Args:
            residue_embeddings: [batch_size, seq_len, input_dim]
            valid_mask: [batch_size, seq_len], 1 for valid residues, 0 for padding/special tokens
            return_attention_weights: Whether to return attention weights for visualization
            
        Returns:
            Dictionary containing:
                - logits: [batch_size, seq_len, num_site_types]，模型经过温度缩放、掩码后最终的原始输出分数。每个残基对于每种位点类型（如背景、结合、催化等）都有一个logit值。这个张量会直接输入到损失函数（如交叉熵损失）中，用于计算主任务的损失
                - prompt_similarities: [batch_size, seq_len, num_site_types]，应用温度缩放（temperature scaling）之前的、残基嵌入与prompt向量之间的直接相似度得分
                - prompt_regularization_loss: scalar (averaged over batch)
                - prompt_diversity_loss: scalar (averaged over batch, if enabled)
                - attention_weights: [batch_size, seq_len, num_site_types] (if requested)
        """
        batch_size, seq_len, _ = residue_embeddings.shape
        
        # Apply layer normalization if enabled
        if self.layer_norm is not None:
            residue_embeddings = self.layer_norm(residue_embeddings)
        
        # Apply dropout
        residue_embeddings = self.dropout(residue_embeddings)
        
        results = {}
        
        # Apply L2 normalization to prompts if enabled
        prompts = self.prompts
        if self.prompt_l2_norm:
            prompts = F.normalize(prompts, dim=-1)
        
        if self.use_multi_head_attention:
            # Use multi-head attention for prompt matching
            # Treat prompts as keys/values and residue embeddings as queries
            prompts_expanded = prompts.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_site_types, input_dim]
            
            # Apply attention: residue embeddings attend to prompts
            attn_output, attn_weights = self.prompt_attention(
                query=residue_embeddings,  # [batch_size, seq_len, input_dim]
                key=prompts_expanded,      # [batch_size, num_site_types, input_dim]  
                value=prompts_expanded,    # [batch_size, num_site_types, input_dim]
                need_weights=return_attention_weights
            )
            
            # Project attention output to site type logits
            logits = torch.einsum('bld,kd->blk', attn_output, prompts)
            
            if return_attention_weights:
                results['attention_weights'] = attn_weights
                
        else:
            # Direct similarity computation
            prompt_similarities = self.compute_prompt_similarity(residue_embeddings, prompts)
            results['prompt_similarities'] = prompt_similarities
            
            # Apply temperature scaling with clamping
            clamped_temperature = self._get_clamped_temperature()
            logits = prompt_similarities / clamped_temperature
        
        # Apply residual connection if enabled
        if self.use_residual_connection:
            residual_logits = self.residual_proj(residue_embeddings)
            logits = logits + residual_logits
        
        # Mask invalid positions if provided
        if valid_mask is not None:
            # Expand mask to match logits dimensions
            mask_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, self.num_site_types)
            logits = logits.masked_fill(~mask_expanded, float('-inf'))
        
        results['logits'] = logits
        
        # Compute regularization losses (already averaged in the methods)
        results['prompt_regularization_loss'] = self.prompt_regularization * self.compute_prompt_regularization_loss()
        
        if self.use_prompt_diversity_loss:
            results['prompt_diversity_loss'] = self.diversity_loss_weight * self.compute_prompt_diversity_loss()
        
        return results
    


    def get_prompt_interpretations(self) -> Dict[str, torch.Tensor]:
        """
        Get prompt vectors for interpretation and visualization.
        
        Returns:
            Dictionary mapping site type names to prompt vectors
        """
        interpretations = {}
        for i, site_name in enumerate(self.site_type_names):
            interpretations[site_name] = self.prompts[i].detach()
        return interpretations
    


    def compute_prompt_similarities_to_residues(
        self, 
        residue_embeddings: torch.Tensor, 
        site_types: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute similarities between specific prompts and residue embeddings for analysis.
        
        Args:
            residue_embeddings: [batch_size, seq_len, input_dim]
            site_types: List of site type names to compute similarities for
            
        Returns:
            Dictionary mapping site type names to similarity tensors
        """
        if site_types is None:
            site_types = self.site_type_names
            
        similarities = {}
        for site_name in site_types:
            if site_name in self.site_type_names:
                site_idx = self.site_type_names.index(site_name)
                prompt = self.prompts[site_idx:site_idx+1]  # [1, input_dim]
                similarity = self.compute_prompt_similarity(residue_embeddings, prompt)
                similarities[site_name] = similarity.squeeze(-1)  # [batch_size, seq_len]
        
        return similarities



    def analyze_loss_characteristics(self) -> Dict[str, Any]:
        """
        分析prompt损失的特征，证明它们不是batch级别的损失。
        
        Returns:
            分析结果字典，包含损失的性质说明
        """
        analysis = {}
        
        # 分析prompt参数的形状
        analysis['prompt_shape'] = list(self.prompts.shape)  # [num_site_types, input_dim]
        analysis['has_batch_dimension'] = False  # prompt参数不包含batch维度
        
        # 计算regularization loss并分析
        reg_loss = self.compute_prompt_regularization_loss()
        analysis['regularization_loss'] = {
            'value': reg_loss.item(),
            'shape': list(reg_loss.shape),  # 标量
            'description': 'L2 regularization on prompt parameters (averaged over all parameters)',
            'depends_on_batch': False  # 不依赖于batch
        }
        
        # 计算diversity loss并分析  
        if self.use_prompt_diversity_loss:
            div_loss = self.compute_prompt_diversity_loss()
            analysis['diversity_loss'] = {
                'value': div_loss.item(),
                'shape': list(div_loss.shape),  # 标量
                'description': f'Diversity loss using {self.diversity_loss_type} metric (averaged over prompt pairs)',
                'depends_on_batch': False  # 不依赖于batch
            }
        
        return analysis
    


    def compute_prompt_regularization_loss(self) -> torch.Tensor:
        """
        计算L2正则化损失，现在返回所有参数的平均值以提高数值稳定性。
        
        注意：这不是batch级别的平均损失，而是参数级别的正则化损失。
        prompt参数的形状是 [num_site_types, input_dim]，不包含batch维度。
        
        Returns:
            reg_loss: 标量张量，所有prompt参数的平均平方值
        """
        # 计算所有prompt参数的平均平方值（而不是总和）
        # 这提供了更好的数值稳定性和可解释性
        total_params = self.num_site_types * self.input_dim
        reg_loss = torch.sum(self.prompts ** 2) / total_params
        
        return reg_loss


    
    def compute_prompt_diversity_loss(self) -> torch.Tensor:
        """
        计算多样性损失，现在返回所有prompt对的平均值以提高数值稳定性。
        
        注意：这不是batch级别的平均损失，而是prompt向量间的相似性正则化损失。
        prompt参数的形状是 [num_site_types, input_dim]，不包含batch维度。
        
        Returns:
            diversity_loss: 标量张量，所有prompt对的平均多样性损失
        """
        if self.diversity_loss_type == "cosine":
            # 余弦相似性基础的多样性损失
            prompt_norm = torch.norm(self.prompts, dim=-1, keepdim=True)
            normalized_prompts = self.prompts / (prompt_norm + self.eps)
            
            # 余弦相似性矩阵 [num_site_types, num_site_types]
            cosine_sim_matrix = torch.mm(normalized_prompts, normalized_prompts.t())
            
            # 掩码对角线元素（自相似性）
            num_pairs = self.num_site_types * (self.num_site_types - 1)
            mask = torch.eye(self.num_site_types, device=self.prompts.device)
            off_diagonal_sims = cosine_sim_matrix * (1 - mask)
            
            # 平均多样性损失（对所有prompt对求平均，而不是求和）
            diversity_loss = torch.sum(torch.abs(off_diagonal_sims)) / num_pairs
            
        elif self.diversity_loss_type == "euclidean":
            # 欧几里得距离基础的多样性损失
            diversity_loss = 0.0
            num_pairs = 0
            for i in range(self.num_site_types):
                for j in range(i + 1, self.num_site_types):
                    distance = torch.norm(self.prompts[i] - self.prompts[j])
                    # 鼓励更大的距离（最小化负距离）
                    diversity_loss += torch.exp(-distance)
                    num_pairs += 1
            
            diversity_loss = diversity_loss / num_pairs if num_pairs > 0 else diversity_loss
            
        else:
            raise ValueError(f"Unknown diversity loss type: {self.diversity_loss_type}")
        
        return diversity_loss



class PairBiasedAttentionLayer(nn.Module):
    """
    一个自注意力层，使用成对特征（Pair Features）作为注意力偏差（Attention Bias）。
    灵感来源于 AlphaFold2 Evoformer 中的行/列注意力机制。
    """
    def __init__(self, node_dim, pair_dim, num_heads, dropout=0.1):
        super().__init__()
        assert node_dim % num_heads == 0, "节点维度必须能被注意力头数整除"
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 用于将节点特征投影到 Q, K, V
        self.to_qkv = nn.Linear(node_dim, node_dim * 3, bias=False)
        # 用于将成对特征投影到注意力偏差
        self.to_bias = nn.Linear(pair_dim, num_heads, bias=False)
        # 输出投影
        self.to_out = nn.Linear(node_dim, node_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, pair_features, mask=None):
        """
        Args:
            nodes (torch.Tensor): 节点特征，形状为 (B, L, D_node)
            pair_features (torch.Tensor): 成对特征，形状为 (B, L, L, D_pair)
            mask (torch.Tensor): 有效残基的掩码，形状为 (B, L)
        
        Returns:
            torch.Tensor: 更新后的节点特征，形状为 (B, L, D_node)
        """
        B, L, _ = nodes.shape

        # 1. 投影到 Q, K, V 并重塑以适应多头注意力
        qkv = self.to_qkv(nodes).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2), qkv) # -> (B, H, L, D_head)

        # 2. 计算注意力 logits
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # (B, H, L, L)

        # 3. 计算并添加成对特征偏差
        pair_bias = self.to_bias(pair_features) # (B, L, L, H)
        pair_bias = pair_bias.permute(0, 3, 1, 2) # -> (B, H, L, L)
        dots = dots + pair_bias

        # 4. 应用掩码
        if mask is not None:
            # mask: (B, L) -> (B, 1, 1, L)
            # 这样广播后，填充位置的列（key）的注意力分数会变为负无穷
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask.view(B, 1, 1, L)
            dots.masked_fill_(~mask, mask_value)

        # 5. 计算注意力权重并应用
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # 6. 聚合 V 值
        out = torch.einsum('bhij,bhjd->bhid', attn, v) # (B, H, L, D_head)
        out = out.transpose(1, 2).reshape(B, L, -1) # -> (B, L, D_node)

        # 7. 输出投影
        return self.to_out(out)



class PairBiasedAttentionPredictor(nn.Module):
    """
    一个新的预测头，它使用多个堆叠的 PairBiasedAttentionLayer 来整合节点和成对特征，
    以进行最终的活性位点预测。
    """
    def __init__(self, node_dim, pair_dim, hidden_dim, output_dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(node_dim),
                PairBiasedAttentionLayer(node_dim, pair_dim, num_heads, dropout),
                nn.LayerNorm(node_dim),
                nn.Sequential( # FeedForward
                    nn.Linear(node_dim, node_dim * 4),
                    GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(node_dim * 4, node_dim)
                )
            ]))
        
        # 最终的分类头
        self.final_head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, node_features, pair_features, mask=None):
        """
        Args:
            node_features (torch.Tensor): 节点特征，形状为 (B, L, D_node)
            pair_features (torch.Tensor): 成对特征，形状为 (B, L, L, D_pair)
            mask (torch.Tensor): 有效残基的掩码，形状为 (B, L)
        
        Returns:
            torch.Tensor: 分类 logits，形状为 (B, L, D_out)
        """
        x = node_features

        # 通过多个交互层
        for norm1, attn, norm2, ff in self.layers:
            # 残差连接
            x = x + attn(norm1(x), pair_features, mask)
            x = x + ff(norm2(x))

        # 分类
        logits = self.final_head(x)
        return logits



class PhiGnetCouplingProcessor(nn.Module):
    """
    A module to preprocess a batch of coupling matrices for sparse batching.
    It can be configured to produce either binary or continuous edge weights.
    """
    def __init__(self, cut_thresh: float = 0.2, use_binary_weights: bool = False, normalize_before_threshold: bool = False):
        """
        Args:
            cut_thresh (float): The threshold to determine edge existence.
            use_binary_weights (bool): If True, edge weights will be 1.0 for all existing edges.
                                       If False, edge weights will be the continuous coupling scores from the matrix.
            normalize_before_threshold (bool): If True, applies min-max normalization to scale the coupling matrix
                                               to the [0, 1] range before applying the cut_thresh.
                                               Defaults to False to maintain backward compatibility.
        """
        super().__init__()
        self.cut_thresh = cut_thresh
        self.use_binary_weights = use_binary_weights
        self.normalize_before_threshold = normalize_before_threshold



    def forward(self, coupling_matrices: List[torch.Tensor], num_residues: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a list of coupling matrices into a sparse format for the whole batch.

        Args:
            coupling_matrices (List[torch.Tensor]): A list of 2D tensors of shape [valid_len_i, valid_len_i].
            num_residues (torch.Tensor): A 1D tensor containing the valid length for each protein.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - edge_index (torch.Tensor): Sparse edge indices for the giant graph, shape [2, |E_total|].
                - edge_weight (torch.Tensor): Weights for each edge, shape [|E_total|,].
                - batch_index (torch.Tensor): A vector mapping each node to its original graph index, shape [|V_res|,].
        """
        edge_indices, edge_weights = [], []
        batch_indices = []
        node_offset = 0

        for i, matrix in enumerate(coupling_matrices):
            device = matrix.device
            valid_len = num_residues[i].item()

            processed_matrix = matrix

            # If normalization is enabled, perform min-max scaling on the matrix.
            if self.normalize_before_threshold:
                min_val = torch.min(matrix)
                max_val = torch.max(matrix)
                denominator = max_val - min_val
                
                # Add a small epsilon for numerical stability, avoiding division by zero
                # if the matrix contains only a single unique value.
                if denominator > 1e-8:
                    processed_matrix = (matrix - min_val) / (denominator + 1e-8)
                else:
                    # If the matrix is flat, the normalized result is a zero matrix.
                    print(f"Warning: Coupling matrix for graph {i} is flat. Normalized matrix will be all zeros.")
                    processed_matrix = torch.zeros_like(matrix)

            # Create a boolean mask for edges above the threshold
            adj_mask = processed_matrix > self.cut_thresh
            
            # Get the indices of the true values (edges) from the mask
            current_edge_index = adj_mask.nonzero(as_tuple=False).t() # Shape: [2, num_edges_i]

            # Decide how to assign edge weights based on the `use_binary_weights` flag.
            if self.use_binary_weights:
                # Assign a weight of 1.0 to all existing edges (binary case).
                num_edges = current_edge_index.shape[1]
                current_edge_weight = torch.ones(num_edges, dtype=torch.float32, device=device)
            else:
                # Get the continuous weights for these edges from the coupling matrix.
                current_edge_weight = processed_matrix[current_edge_index[0], current_edge_index[1]]

            # Adjust indices by adding the node offset from previous graphs
            edge_indices.append(current_edge_index + node_offset)
            edge_weights.append(current_edge_weight)
            batch_indices.append(torch.full((valid_len,), i, dtype=torch.long, device=device))
            
            node_offset += valid_len
        
        # Concatenate all lists into single tensors
        if not edge_indices:
            raise ValueError("No edges found in the provided coupling matrices with the given threshold.")
        else:
            final_edge_index = torch.cat(edge_indices, dim=1)
            final_edge_weight = torch.cat(edge_weights, dim=0)

        final_batch_index = torch.cat(batch_indices, dim=0)

        return final_edge_index, final_edge_weight, final_batch_index



class PhiGnetGraphConv(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, use_bias: bool = True):
        """
        Initializes the PhiGnet-style GCN block.

        Args:
            input_dim (int): Dimension of the input node features.
            hidden_dim (int): Dimension of the hidden and output features for each GCN layer.
            num_layers (int): The number of GCN layers to stack. PhiGnet uses 3.
            use_bias (bool): Whether to use a bias term in GCN layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # Input projection to match GCN hidden dimensions
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Create a stack of GCN layers
        for _ in range(num_layers):
            self.layers.append(
                GCNConv(hidden_dim, hidden_dim, bias=use_bias)
            )



    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-layer GCN.

        Args:
            x (torch.Tensor): Node features for the entire batch, shape [|V_res|, input_dim].
            edge_index (torch.Tensor): Edge indices for the batch, shape [2, |E_total|].
            edge_weight (torch.Tensor): Edge weights for the batch, shape [|E_total|,].

        Returns:
            torch.Tensor: Concatenated output node features from all GCN layers.
                          Shape: [|V_res|, hidden_dim * num_layers].
        """
        # Initial projection of input features
        x = self.input_proj(x)
        x = F.relu(x)

        # Store outputs from each layer for final concatenation
        layer_outputs = []

        # Process through the stack of GCN layers
        for i in range(self.num_layers):
            # Each GCN layer operates on the output of the previous layer
            x = self.layers[i](x, edge_index, edge_weight)
            x = F.relu(x)
            layer_outputs.append(x)
        
        # Concatenate the outputs of all layers, mimicking PhiGnet's structure
        final_output = torch.cat(layer_outputs, dim=-1)
        
        return final_output



class GraphECFeatureCalculator(nn.Module):

    def __init__(self, num_rbf=16, rbf_min=0., rbf_max=20.):
        super().__init__()
        self.num_rbf = num_rbf
        self.rbf_min = rbf_min
        self.rbf_max = rbf_max
        self.register_buffer('D_mu', torch.linspace(self.rbf_min, self.rbf_max, self.num_rbf))
        self.register_buffer('D_sigma', torch.tensor((self.rbf_max - self.rbf_min) / self.num_rbf))



    def _rbf(self, D):
        """Vectorized Radial Basis Function embedding."""
        D_expand = D.unsqueeze(-1)
        return torch.exp(-((D_expand - self.D_mu) / self.D_sigma) ** 2)



    def _get_distances_and_directions(self, X, edge_index):
        # This helper function is already correct from the previous step and needs no changes.
        atom_N, atom_Ca, atom_C, atom_O, atom_R = X.unbind(dim=1)
        node_pair_vectors = torch.stack([
            atom_Ca - atom_N, atom_Ca - atom_C, atom_Ca - atom_O,
            atom_N - atom_C, atom_N - atom_O, atom_O - atom_C,
            atom_R - atom_N, atom_R - atom_Ca, atom_R - atom_C, atom_R - atom_O
        ], dim=1)
        node_dist = self._rbf(node_pair_vectors.norm(dim=-1)).view(X.size(0), -1)
        X_i = X[edge_index[0]]
        X_j = X[edge_index[1]]
        edge_pair_vectors = X_i.unsqueeze(2) - X_j.unsqueeze(1)
        edge_dist = self._rbf(edge_pair_vectors.norm(dim=-1)).view(edge_index.size(1), -1)
        u = F.normalize(atom_Ca - atom_N, dim=-1)
        v = F.normalize(atom_C - atom_Ca, dim=-1)
        b = F.normalize(u - v, dim=-1)
        n = F.normalize(torch.cross(u, v), dim=-1)
        local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)
        node_dir_vectors = F.normalize(X[:, [0,2,3,4]] - atom_Ca.unsqueeze(1), dim=-1)
        node_direction = torch.matmul(node_dir_vectors, local_frame).view(X.size(0), -1)
        source_nodes_idx = edge_index[0]
        dest_nodes_idx = edge_index[1]
        t1 = F.normalize(X[source_nodes_idx] - atom_Ca[dest_nodes_idx].unsqueeze(1), dim=-1)
        vec1 = torch.matmul(t1, local_frame[dest_nodes_idx]).view(edge_index.size(1), -1)
        t2 = F.normalize(X[dest_nodes_idx] - atom_Ca[source_nodes_idx].unsqueeze(1), dim=-1)
        vec2 = torch.matmul(t2, local_frame[source_nodes_idx]).view(edge_index.size(1), -1)
        edge_direction = torch.cat([vec1, vec2], dim=-1)
        r = torch.matmul(local_frame[dest_nodes_idx].transpose(-1, -2), local_frame[source_nodes_idx])
        diag = torch.diagonal(r, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)))
        _R = lambda i,j: r[:,i,j]
        signs = torch.sign(torch.stack([_R(2,1) - _R(1,2), _R(0,2) - _R(2,0), _R(1,0) - _R(0,1)], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        edge_orientation = F.normalize(Q, dim=-1)
        return node_dist, node_direction, edge_dist, edge_direction, edge_orientation
    


    def forward(self, graph, atom_coords, edge_index):
        # --- Positional Embeddings (Equivalent) ---
        d = edge_index[0] - edge_index[1]
        frequency = torch.exp(
            torch.arange(0, 16, 2, dtype=torch.float32, device=edge_index.device)
            * -(torch.log(torch.tensor(10000.0)) / 16)
        )
        angles = d.unsqueeze(-1) * frequency
        pos_embeddings = torch.cat((torch.cos(angles), torch.sin(angles)), -1)

        # --- Angles (MODIFIED for 100% Equivalence) ---
        node_angles_list = []
        node_splits = graph.num_nodes.tolist()
        node_indices = torch.cumsum(torch.tensor([0] + node_splits, device=graph.device), dim=0)

        for i in range(graph.batch_size):
            start, end = node_indices[i], node_indices[i+1]
            protein_atom_coords = atom_coords[start:end]
            num_residues = protein_atom_coords.shape[0]

            if num_residues < 2:
                node_angles_list.append(torch.zeros(num_residues, 12, device=atom_coords.device, dtype=atom_coords.dtype))
                continue
            
            X = torch.reshape(protein_atom_coords[:, :3], [3 * num_residues, 3])
            dX = X[1:] - X[:-1]
            U = F.normalize(dX, dim=-1)

            if U.shape[0] < 3:
                node_angles_list.append(torch.zeros(num_residues, 12, device=atom_coords.device, dtype=atom_coords.dtype))
                continue
                
            u_2, u_1, u_0 = U[:-2], U[1:-1], U[2:]
            n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
            n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

            # Dihedral angles (psi, omega, phi)
            cosD_dihedral = torch.sum(n_2 * n_1, -1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            D_dihedral = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD_dihedral)
            D_dihedral = F.pad(D_dihedral, [1, 2])
            D_dihedral = torch.reshape(D_dihedral, [-1, 3])
            
            # Bond angles (alpha, beta, gamma)
            cosD_bond = (u_2 * u_1).sum(-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            D_bond = torch.acos(cosD_bond)
            D_bond = F.pad(D_bond, [1, 2])
            D_bond = torch.reshape(D_bond, [-1, 3])
            
            # --- START CORRECTION ---
            # The original code concatenates cos and sin separately (grouped),
            # while the previous GPU version did it interleaved.
            # We now replicate the grouped concatenation.
            
            # Original: dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1) -> [cos(psi), cos(om), cos(phi)], [sin(psi), sin(om), sin(phi)]
            dihedral_cos = torch.cos(D_dihedral) # Shape: (L, 3)
            dihedral_sin = torch.sin(D_dihedral) # Shape: (L, 3)
            dihedral = torch.cat([dihedral_cos, dihedral_sin], dim=1) # Shape: (L, 6)

            # Original: bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)
            bond_angles_cos = torch.cos(D_bond) # Shape: (L, 3)
            bond_angles_sin = torch.sin(D_bond) # Shape: (L, 3)
            bond_angles = torch.cat([bond_angles_cos, bond_angles_sin], dim=1) # Shape: (L, 6)
            
            # Final node angles are the concatenation of these two parts
            node_angles = torch.cat((dihedral, bond_angles), 1) # Shape: (L, 12)
            # --- END CORRECTION ---

            node_angles_list.append(node_angles)
        
        node_angles = torch.cat(node_angles_list, dim=0)

        # --- Other Features (Already Equivalent) ---
        node_dist, node_direction, edge_dist, edge_direction, edge_orientation = self._get_distances_and_directions(atom_coords, edge_index)

        # --- Final Concatenation ---
        geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)
        geo_edge_feat = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)

        return geo_node_feat, geo_edge_feat
    


class GNNLayer(nn.Module):
    """
    define GNN layer for subsequent computations
    """
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E



class EdgeMLP(nn.Module):
    """
    define MLP operation for edge updates
    """
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E



class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V
    


class GraphECEncoder(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 4, drop_rate: float = 0.2):
        """
        Initializes the GraphECEncoder.

        Args:
            node_in_dim (int): Dimension of the input node features.
            edge_in_dim (int): Dimension of the input edge features.
            hidden_dim (int): The hidden dimension of the GNN layers.
            output_dim (int): The final dimension of the output node feature vectors.
            num_layers (int): The number of GNN layers to stack.
            drop_rate (float): The dropout rate used within the GNN layers and output head.

            Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, seq_in=False, num_layers=num_layers, drop_rate=dropout)
        """
        super(GraphECEncoder, self).__init__()

        # --- Input Embedding Layers (from original Graph_encoder) ---
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # --- Core GNN Layers (from original Graph_encoder) ---
        self.layers = nn.ModuleList(
            GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers)
        )

        # --- New Output Head (replaces FC layers in GraphEC_AS) ---
        # This projects the GNN's hidden features to the final output dimension.
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # Analogous to the first FC layer
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, output_dim)  # Projects to the final feature vector
        )

        self._initialize_weights()



    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, h_V: torch.Tensor, edge_index: torch.Tensor, h_E: torch.Tensor, batch_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            h_V (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity in COO format.
            h_E (torch.Tensor): Input edge features.
            batch_id (torch.Tensor): Batch indicator for each node.

        Returns:
            torch.Tensor: The output node feature vectors, shape [num_nodes, output_dim].
        """
        # --- GNN Core Processing (from original Graph_encoder.forward) ---
        # 1. Embed and normalize node and edge features
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        # 2. Process through GNN layers
        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        
        # --- Output Projection ---
        # 3. Pass the final node hidden states through the output head
        output_features = self.output_head(h_V)
        
        return output_features



class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, positional_number=5, dropout=0.1):
        """
        input:
            positional_number: 25th equation in the paper, Total number of discrete categories of interatomic spatial distances
        """
        super(MultiHeadAttention, self).__init__()
        self.p_k = positional_number
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        if self.p_k != 0:
            self.relative_k = nn.Parameter(torch.randn(self.p_k, self.d_k))
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(d_model, d_model))
        self.gating = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

    def one_hot_embedding(self, labels):
        y = torch.eye(self.p_k, device=labels.device)
        return y[labels]

    def forward(self, src, tgt, gpm, src_mask=None, tgt_mask=None):

        bs, atom_size = src.size(0), src.size(1)
        src = self.layer_norm(src)
        tgt = self.layer_norm(tgt)
        k = self.k_linear(src)
        q = self.q_linear(tgt)
        v = self.v_linear(tgt)
        k1 = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q1 = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v1 = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        attn1 = torch.matmul(q1, k1.permute(0, 1, 3, 2))


        if (self.p_k == 0) or (gpm is None):
            attn = attn1 / math.sqrt(self.d_k)
        elif gpm is not None:
            gpms = self.one_hot_embedding(
                gpm.unsqueeze(1).repeat(1, self.h, 1, 1))
            attn2 = torch.matmul(q1, self.relative_k.transpose(0, 1))
            attn2 = torch.matmul(gpms, attn2.unsqueeze(-1)).squeeze(-1)
            attn = (attn1 + attn2) / math.sqrt(self.d_k)

        mask_value = torch.finfo(attn.dtype).min
        if (src_mask is not None) and (tgt_mask is None):
            src_mask = src_mask.bool()
            src_mask = src_mask.unsqueeze(1).repeat(1, src_mask.size(-1), 1)
            src_mask = src_mask.unsqueeze(1).repeat(1, attn.size(1), 1, 1)
            attn_mask = src_mask
            attn[~attn_mask] = mask_value
        elif (src_mask is not None) and (tgt_mask is not None):
            src_mask = src_mask.bool()
            tgt_mask = tgt_mask.bool()
            attn_mask = tgt_mask.unsqueeze(1).repeat(1, src_mask.size(-1), 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, attn.size(1), 1, 1)
            attn = attn.permute(0, 1, 3, 2)
            attn[~attn_mask] = mask_value

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout1(attn)
        v1 = v.view(bs, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        output = torch.matmul(attn, v1)

        output = output.transpose(1, 2).contiguous().view(
            bs, -1, self.d_model).squeeze(-1)
        # gate self attention
        output = self.to_out(output * self.gating(src).sigmoid())
        return self.dropout2(output), attn, attn_mask
    

    
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))



# TODO: self attention 和 cross attention 交叉进行
class GlobalMultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 heads=8,
                 positional_number=5,
                 cross_attn_h_rate=1,
                 dropout=0.1,):
        super(GlobalMultiHeadAttentionLayer, self).__init__()

        self.cross_attn_h_rate = cross_attn_h_rate

        self.self_attn = MultiHeadAttention(heads, d_model, positional_number,
                                            dropout)
        self.cross_attn = MultiHeadAttention(heads, d_model, 0,
                                             dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = GELU()

    def forward(self, src, tgt, rpm=None, src_mask=None, tgt_mask=None):
        """
        If you want to merge B's information into A, A should be src and B should be tgt.

        output:
            src: updated src features
            att_score_s: self-attention score
            attn_mask_s: self-attention mask
            att_score_c: cross-attention score
            attn_mask_c: cross-attention mask
        """

        src_m_s, att_score_s, attn_mask_s = self._sa_block(
            self.norm1(src), rpm, src_mask)

        src = src + src_m_s

        src_m_c, att_score_c, attn_mask_c = self._cra_block(
            self.norm2(src), tgt, src_mask, tgt_mask)

        src = src * self.cross_attn_h_rate + src_m_c  # self.cross_attn_h_rate 控制交叉注意力机制自身的比例
        src = src + self._ff_block(self.norm3(src))

        return src, att_score_s, attn_mask_s, att_score_c, attn_mask_c

    def _sa_block(self, x, rpm, mask):
        m, att_score, attn_mask = self.self_attn(x, x, rpm, mask)
        return self.dropout1(m), att_score, attn_mask

    def _cra_block(self, src, tgt, src_mask, tgt_mask):
        src_m, att_score, attn_mask = self.cross_attn(src, tgt, None, src_mask,
                                                      tgt_mask)
        return self.dropout2(src_m), att_score, attn_mask

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



class GlobalMultiHeadAttention(nn.Module):
    """
    atom distance self-attention module and substrate-product cross-attention module
    or
    enzyme self-attention module and enzyme-substrate cross-attention module
    """
    def __init__(self,
                 d_model,
                 heads=8,
                 n_layers=3,
                 positional_number=5,
                 cross_attn_h_rate=0.1,
                 dropout=0.1):
        super(GlobalMultiHeadAttention, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        layer_stack = []

        for _ in range(n_layers):

            layer_stack.append(
                GlobalMultiHeadAttentionLayer(
                    d_model=d_model,
                    heads=heads,
                    positional_number=positional_number,
                    cross_attn_h_rate=cross_attn_h_rate,
                    dropout=dropout))

        self.layers = nn.ModuleList(layer_stack)

    def forward(self, src, tgt, rpm=None, src_mask=None, tgt_mask=None):
        """
        input:
            src (Tensor): source sequence features, shape is [batch_size, seq_len, d_model]
            tgt (Tensor): target sequence features, shape is [batch_size, seq_len, d_model]
            rpm (Tensor, optional): relative position matrix, used to encode the spatial distance relationship between atoms in the molecule, shape is [batch_size, seq_len, seq_len]
            src_mask (Tensor, optional): source sequence mask, used to mark which positions are valid atoms and which are fillers, shape is [batch_size, seq_len]
            tgt_mask (Tensor, optional): target sequence mask, used to mark which positions are valid atoms and which are fillers, shape is [batch_size, seq_len]
        
        output:
            src (Tensor): updated source sequence features, shape is [batch_size, seq_len, d_model]
            self_att_scores (dict): self-attention scores for each layer, keys are layer indices and values are attention scores, shape is [batch_size, heads, seq_len, seq_len]
            self_attn_mask (Tensor): self-attention mask for the last layer, shape is [batch_size, heads, seq_len, seq_len]
            cross_att_scores (dict): cross-attention scores for each layer, keys are layer indices and values are attention scores, shape is [batch_size, heads, seq_len, seq_len]
            cross_attn_mask (Tensor): cross-attention mask for the last layer, shape is [batch_size, heads, seq_len, seq_len]
        """
        self_att_scores = {}
        cross_att_scores = {}

        for n in range(self.n_layers):

            src, att_score_s, attn_mask_s, att_score_c, attn_mask_c = self.layers[
                n](src, tgt, rpm, src_mask, tgt_mask)
            self_att_scores[n] = att_score_s
            cross_att_scores[n] = att_score_c
        self_attn_mask = attn_mask_s
        cross_attn_mask = attn_mask_c
        return src, self_att_scores, self_attn_mask, cross_att_scores, cross_attn_mask



# =========================================== Top@Utilities Functions ====================================================== 
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger



def get_step_based_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 10000,
    warmup_steps: int = 500,
    min_lr: float = 1e-6,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create step-based learning rate scheduler for better training control.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("cosine", "linear", "step", "exponential")
        total_steps: Total number of training steps (optimizer updates)
        warmup_steps: Number of warmup steps
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Learning rate scheduler
    """

    
    initial_lr = optimizer.param_groups[0]['lr']
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup phase: linear increase from 0.1 to 1.0
            return 0.1 + 0.9 * step / warmup_steps
        else:
            # Main phase
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            
            if scheduler_type == "cosine":
                # Cosine annealing
                return min_lr / initial_lr + (1 - min_lr / initial_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            elif scheduler_type == "linear":
                # Linear decay
                return min_lr / initial_lr + (1 - min_lr / initial_lr) * (1 - progress)
            elif scheduler_type == "constant":
                # Constant learning rate after warmup
                return 1.0
            elif scheduler_type == "exponential":
                # Exponential decay
                gamma = kwargs.get('gamma', 0.95)
                return gamma ** (step - warmup_steps)
            elif scheduler_type == "polynomial":
                # Polynomial decay
                power = kwargs.get('power', 1.0)
                return min_lr / initial_lr + (1 - min_lr / initial_lr) * (1 - progress) ** power
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return LambdaLR(optimizer, lr_lambda)



def create_loss_function(
    loss_type: str = "ce",
    class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
    pos_weight: Optional[Union[torch.Tensor, List[float]]] = None,
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        loss_type: Type of loss ("ce", "bce", "focal")
        class_weights: Class weights for imbalanced datasets，类别权重（用于处理类别不平衡），一般用于交叉熵损失。
        pos_weight: Positive class weight for BCE，正样本权重（用于二元交叉熵），适合正负样本极度不均衡的情况。
        label_smoothing: Label smoothing factor，标签平滑参数（用于交叉熵损失），可以缓解过拟合。
        
    Returns:
        Loss function
    """
    # Convert class_weights to torch.Tensor if it's a list
    if class_weights is not None and isinstance(class_weights, (list, tuple)):
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Convert pos_weight to torch.Tensor if it's a list
    if pos_weight is not None and isinstance(pos_weight, (list, tuple)):
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    


def log_metrics(
    metrics: Dict[str, float],
    epoch: int,
    mode: str = "train",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log training/validation metrics.
    
    Args:
        metrics: Dictionary of metric name -> value
        epoch: Current epoch (0-based internally)
        mode: Mode ("train", "val", "test")
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Special formatting for alignment_loss to show scientific notation
    formatted_metrics = []
    for k, v in metrics.items():
        if 'alignment_loss' in k:
            formatted_metrics.append(f"{k}: {v:.6e}")
        else:
            formatted_metrics.append(f"{k}: {v:.4f}")
    
    metrics_str = " | ".join(formatted_metrics)
    # Display epoch as 1-based for user-friendly output
    logger.info(f"Epoch {epoch + 1:3d} [{mode.upper()}] - {metrics_str}")



def validate_tensor(tensor, name="tensor", check_finite=True, check_range=None, eps=1e-8):
    """
    验证张量的数值稳定性
    
    Args:
        tensor: 待验证的张量
        name: 张量名称，用于错误信息
        check_finite: 是否检查有限性（NaN/Inf）
        check_range: 可选的数值范围 (min, max)
        eps: 用于范围检查的容差
    
    Returns:
        bool: 张量是否有效
    """
    if tensor is None:
        print(f"Tensor validation issues for {name}: {name} is None")
        return False
    
    issues = []
    
    if check_finite:
        if torch.isnan(tensor).any():
            issues.append(f"{name} contains NaN")
        if torch.isinf(tensor).any():
            issues.append(f"{name} contains Inf")
    
    if check_range is not None:
        min_val, max_val = check_range
        actual_min = tensor.min().item()
        actual_max = tensor.max().item()
        if actual_min < min_val - eps or actual_max > max_val + eps:
            issues.append(f"{name} out of range [{min_val}, {max_val}]: [{actual_min:.4f}, {actual_max:.4f}]")
    
    if issues:
        print(f"Tensor validation issues for {name}: {'; '.join(issues)}")
        return False
    
    return True



def gradient_clipping_with_monitoring(model, max_norm=1.0):
    """
    带监控的梯度裁剪
    
    Args:
        model: PyTorch模型
        max_norm: 最大梯度范数
    
    Returns:
        float: 原始梯度范数
    """
    # 计算原始梯度范数
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2)
    
    # 检查梯度中的NaN
    nan_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_count += 1
                print(f"Warning: NaN gradients in {name}")
                param.grad.zero_()
    
    if nan_count > 0:
        print(f"Warning: Found NaN gradients in {nan_count} parameters, zeroed them")
    
    return total_norm.item()



def safe_loss_combination(*losses, weights=None):
    """
    安全地组合多个loss值
    
    Args:
        *losses: 多个loss张量
        weights: 可选的权重列表
    
    Returns:
        torch.Tensor: 组合后的loss
    """
    if not losses:
        print("Warning: No losses provided to combine")
        return torch.tensor(0.0, requires_grad=True)
    
    valid_losses = []
    valid_weights = []
    
    for i, loss in enumerate(losses):
        if loss is not None and torch.isfinite(loss).all():
            valid_losses.append(loss)
            if weights is not None and i < len(weights):
                valid_weights.append(weights[i])
            else:
                valid_weights.append(1.0)
        else:
            print(f"Warning: Invalid loss at index {i}: {loss}, skipping")
    
    if not valid_losses:
        print("Warning: No valid losses found, returning zero")
        return torch.tensor(0.0, requires_grad=True)
    
    # 组合有效的losses
    if len(valid_losses) == 1:
        return valid_losses[0] * valid_weights[0]
    
    total_loss = sum(loss * weight for loss, weight in zip(valid_losses, valid_weights))
    # total_weight = sum(valid_weights)
    
    # if total_weight == 0:
    #     return torch.tensor(0.0, requires_grad=True)
    
    # return total_loss / total_weight
    return total_loss



def pack_residue_feats(packedgraph, residue_feats):
    """
    input:
        packedgraph: [class: torchdrug.data.graph.PackedGraph] object.
        residue_feats: (|V_{res}|, d), |V_res| is the total number of amino acid residues across all proteins in the batch, d is the feature dimension 
    output:
        residue_feats: [batch_size, max_num_residue, node_dim]
        masks: [batch_size, max_num_residue], 1 for valid residue, 0 for padding residue
    """
    num_residues = packedgraph.num_residues.tolist()
    edit_feats = torch.split(residue_feats, num_residues)
    masks = [
        torch.ones(num_residue, dtype=torch.uint8, device=residue_feats.device)
        for num_residue in num_residues
    ]

    # function: torch.nn.utils.rnn.pad_sequence
    padded_feats = pad_sequence(edit_feats, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)
    return padded_feats, masks



def read_model_state(model_save_path):
    model_state_fname = os.path.join(model_save_path, 'model.pth')
    args_fname = os.path.join(model_save_path, 'args.yml')
    # eval_results_fname = os.path.join(model_save_path, 'eval_results.csv')

    model_state = torch.load(model_state_fname,
                             map_location=torch.device('cpu'))
    keys = list(model_state.keys())
    if 'module.' in keys[0]:
        model_state = {k.replace('module.', ''): v for k,v in model_state.items()}
    args = yaml.load(open(args_fname, "r"), Loader=yaml.FullLoader)

    return model_state, args



def load_pretrain_model_state(model, pretrained_state, load_active_net=True):
    model_state = model.state_dict()
    pretrained_state_filter = {}
    extra_layers = []
    different_shape_layers = []
    need_train_layers = []
    for name, parameter in pretrained_state.items():
        if name in model_state and parameter.size() == model_state[name].size():
            pretrained_state_filter[name] = parameter
        elif name not in model_state:
            extra_layers.append(name)
        elif parameter.size() != model_state[name].size():
            different_shape_layers.append(name)
        else:
            pass
    if not load_active_net:
        for name, parameter in model_state.items():
            if 'active_net' in name:
                del pretrained_state_filter[name]
    
    for name, parameter in model_state.items():
        if name not in pretrained_state_filter:
            need_train_layers.append(name)

    model_state.update(pretrained_state_filter)
    model.load_state_dict(model_state)
    
    print('Extra layers:', extra_layers)
    print('Different shape layers:', different_shape_layers)
    print('Need to train layers:', need_train_layers)
    return model



def move_batch_to_device(batch: Any, device) -> Any:
    # Fast path for primitive types and None, which don't need moving.
    if batch is None or isinstance(batch, (str, bytes, int, float, bool)):
        return batch

    # The most common case: a torch.Tensor.
    # Using isinstance is a slight performance optimization over hasattr for this frequent check.
    if isinstance(batch, torch.Tensor):
        try:
            # Use non_blocking for performance gains with pinned memory.
            return batch.to(device, non_blocking=True)
        except TypeError:
            # Fallback for older PyTorch versions or specific tensor types
            # that might not support non_blocking.
            return batch.to(device)

    # Generic handling for any object that implements a `.to()` method (duck-typing).
    # This covers DGL graphs, TorchDrug's PackedGraph, HuggingFace's BatchEncoding, etc.
    # We also ensure it's not a Module, which should not be moved this way.
    if hasattr(batch, "to") and callable(getattr(batch, "to")) and not isinstance(batch, nn.Module):
        try:
            # EAFP (Easier to Ask for Forgiveness than Permission) approach.
            return batch.to(device, non_blocking=True)
        except TypeError:
            # If the '.to' method doesn't support 'non_blocking', try without it.
            return batch.to(device)
        except Exception:
            # If the '.to' method fails for any other reason, return the object as is.
            return batch

    # Recursively handle dictionaries, preserving the dictionary type.
    if isinstance(batch, dict):
        return type(batch)({k: move_batch_to_device(v, device) for k, v in batch.items()})
    
    # Recursively handle lists and tuples, preserving the container type.
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(item, device) for item in batch)

    # Handle NumPy arrays, assuming 'import numpy as np' is present at the top of the file.
    # This check is placed after the main tensor/object checks for optimization.
    if isinstance(batch, np.ndarray):
        return torch.from_numpy(batch).to(device, non_blocking=True)
    
    # If the object type is not recognized, return it unmodified.
    return batch



@torch.no_grad()
def evaluate_ec_prediction(
    reference_protein_features: torch.Tensor,
    query_protein_features: torch.Tensor,
    ec_to_indices_map: Dict[str, List[int]],
    query_ids: List[str],
    top_k: int,
    device: torch.device,
    id_ec_query: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    * 功能：  
        * 评估EC号注释性能，返回宏平均F1分数。
        * 或对未知数据进行EC预测。

    * 一定要确保reference_protein_features和query_protein_features是经过相同的模型或者一组模型处理得到的，否则评估结果没有意义。
    * 输入参数之间必须满足的关系：
        * ec_to_indices_map必须正确对应reference_protein_features
        * query_ids与query_protein_features的第一维顺序必须一致
    Args:
        reference_protein_features (torch.Tensor): 用于构建EC簇中心的参考嵌入 (来自训练集)。
        query_protein_features (torch.Tensor): 需要被评估的查询嵌入 (来自验证集或测试集)。
        ec_to_indices_map (Dict[str, List[int]]): 从EC号到参考嵌入中对应索引的映射。
        id_ec_query (Dict[str, List[str]]): 查询集中蛋白质ID到其真实EC号列表的映射。
            * 如果提供，则计算评估指标。
            * 如果为 None，则仅返回预测结果（用于推理模式）。
        query_ids (List[str]): 按`顺序`排列的查询集蛋白质ID列表，与 `query_protein_features` 的第一维对应，要求query_ids与query_protein_features第一维的顺序严格一致。
        top_k (int): `在CLEAN项目中top_k==10`
            * top_k == 1 时，适用于单标签场景（测试集每个样本只对应一个EC号），仅预测 Top-1；
            * top_k > 1 时，适用于多标签场景，沿用 maximum-separation（预测数由拐点决定，先取最近的 top_k 作为候选，然后在候选内执行 maximum-separation。）。
        device (torch.device): 计算设备。
    Returns:
        metrics (Dict[str, float]): 包含 'ec_f1' 等指标的字典（如果是纯推理模式，则为空字典）。
        predictions (Dict[str, List[str]]): 预测结果字典，{uid: [EC1, EC2, ...]}，列表按概率从高到低排序。
    """
    # 1 计算所有EC的簇中心
    ec_centers = {}
    for ec, indices in ec_to_indices_map.items():
        if indices:
            ec_centers[ec] = reference_protein_features[indices].mean(dim=0)
    
    center_ecs = sorted(ec_centers.keys())
    if not center_ecs:
        return {}
    center_tensors = torch.stack([ec_centers[ec] for ec in center_ecs]).to(device)

    if top_k < 1:   raise ValueError("top_k must be >= 1")
    num_ec_centers = center_tensors.size(0)
    k_candidates = min(int(top_k), num_ec_centers)


    # 2 计算查询嵌入与所有EC簇中心的距离矩阵
    # dist_matrix 的形状: [num_query_proteins, num_ec_centers]
    dist_matrix = torch.cdist(query_protein_features.to(device), center_tensors)


    # 3 对每个查询蛋白应用CLEAN的maximum separation预测策略
    all_pred_ecs = []   # all_pred_ecs 的形状是 list[list[str]]，与 query_ids 一一对应。第 i 个元素是第 i 个查询样本的 多标签预测（可变长）。 e.g., [['1.1.1.1'], ['2.7.1.1', '2.7.1.2']]
    all_pred_probs = []     # 存储每个样本的预测置信度分数列表, e.g., [[0.95], [0.88, 0.85]]
    for i in range(len(query_ids)):
        distances_to_centers = dist_matrix[i]   # 获取当前查询蛋白与所有EC中心的距离
        
        # 找到距离最近的 top-k (这里取10，与CLEAN一致) EC
        # top_k_dists → 最近的k_candidates个距离；top_k_indices → 对应的EC中心索引；
        top_k_dists, top_k_indices = torch.topk(distances_to_centers, k=k_candidates, largest=False)
        
        if top_k == 1:  # 单标签：仅预测 Top-1（不做 maximum-separation）
            pred_indices = top_k_indices[:1]
            pred_dists   = top_k_dists[:1]
        else:           # 多标签：在“前 top_k 候选”内执行 maximum-separation
            # maximum_separation, 计算梯度来找到距离分布中的“拐点”
            dist_lst = top_k_dists.cpu().numpy()
            gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
            sep_lst = np.abs(dist_lst - np.mean(gamma))
            sep_grad = np.abs(sep_lst[:-1] - sep_lst[1:])
            large_grads = np.where(sep_grad > np.mean(sep_grad))    # 找到第一个显著的梯度下降点作为截断点
            if large_grads[0].size > 0:
                max_sep_i = large_grads[0][0] # 取第一个大于均值的梯度位置
            else:
                max_sep_i = 0 # 如果没有显著梯度，默认只取最近的1个
            if max_sep_i >= 5:  # 防止截断点过大，限制在5以内 (CLEAN中的启发式规则)
                max_sep_i = 0
            
            pred_indices = top_k_indices[:max_sep_i + 1]    # 根据截断点获取最终预测的EC列表
            pred_dists = top_k_dists[:max_sep_i + 1]
        
        protein_pred_ecs = [center_ecs[idx] for idx in pred_indices]
        all_pred_ecs.append(protein_pred_ecs)

        pred_dists[pred_dists == 0] = 1e-9
        probs = (1 - torch.exp(-1 / pred_dists)) / (1 + torch.exp(-1 / pred_dists))
        if torch.sum(probs) > 0:    probs = probs / torch.sum(probs)
        all_pred_probs.append(probs.cpu().tolist())
    predictions = {uid: ecs for uid, ecs in zip(query_ids, all_pred_ecs)}


    # 4 准备真实标签并计算分数
    metrics = {}
    if id_ec_query is not None:
        true_labels = [id_ec_query.get(uid, []) for uid in query_ids]
        
        all_label = set()
        for labels in true_labels:      # 仅基于“真值出现过的EC”定义类别全集（与 CLEAN 一致）
            all_label.update(labels)
        
        mlb = MultiLabelBinarizer()
        mlb.fit([list(all_label)])

        true_m = mlb.transform(true_labels)  # true_binarized.shape == (N, C),其中 N 是样本数，C 是 EC 类别数。第 i 行第 j 列为 1 表示“第 i 个样本属于第 j 个 EC”。
        pred_m = mlb.transform(all_pred_ecs)     # pred_binarized.shape == (N, C)

        n_samples, n_classes = true_m.shape
        pred_m_auc = np.zeros((n_samples, n_classes), dtype=float)
        class_to_idx = {cls: idx for idx, cls in enumerate(mlb.classes_)}
        for i, (labels_i, probs_i) in enumerate(zip(all_pred_ecs, all_pred_probs)):
            # 只在 all_label（真值类空间）里存在的 EC 上填分数（与 CLEAN 一致）
            for label, score in zip(labels_i, probs_i):
                idx = class_to_idx.get(label, None)
                if idx is not None:
                    pred_m_auc[i, idx] = score

        metrics['ec_precision'] = precision_score(true_m, pred_m, average='weighted', zero_division=0)
        metrics['ec_recall']    = recall_score(true_m,  pred_m, average='weighted', zero_division=0)
        metrics['ec_f1']        = f1_score(true_m,     pred_m, average='weighted', zero_division=0)
        metrics['ec_roc_auc']   = roc_auc_score(true_m, pred_m_auc, average='weighted')
        metrics['ec_accuracy']  = accuracy_score(true_m, pred_m)

    return metrics, predictions


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)
    

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def extract_clean_state_dict(
    checkpoint: Union[Dict, str], 
    prefix_to_strip: str = "backbone.", 
    ignore_top_keys: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    通用状态字典清洗函数。
    
    Args:
        checkpoint: Checkpoint 字典。
        prefix_to_strip: 需要剥离的主前缀 (如 "backbone.")，用于提取子模型权重。
        ignore_top_keys: 需要移除的【顶层】键名前缀列表 (如 ["proj_head"])。
                         逻辑：只移除以 "key." 开头或完全等于 "key" 的项。
                         不会误伤 "backbone.proj_head" (因为它以 backbone 开头)。
    """
    # 解包嵌套结构
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    
    # 默认忽略 DDP 前缀和统计数据
    if ignore_top_keys is None:
        ignore_top_keys = []
    
    system_ignore = ["total_ops", "total_params"]

    for k, v in state_dict.items():
        # A. 移除 DDP 的 module. 前缀 (放在最前，因为它是 PyTorch 包装产生的)
        if k.startswith("module."):
            k = k[7:]
            
        # B. 精准移除指定的顶层模块 (如 proj_head)
        # 逻辑：只移除 self.proj_head (即键名为 "proj_head.xxx")
        # 绝不移除 self.backbone.proj_head (即键名为 "backbone.proj_head.xxx")
        is_ignored = False
        for ignore_key in (ignore_top_keys + system_ignore):
            # 匹配 "proj_head" 自身 或 "proj_head." 开头
            if k == ignore_key or k.startswith(ignore_key + "."):
                if verbose: print(f"[StateDict] Ignoring key: {k}")
                is_ignored = True
                break
        if is_ignored:
            continue

        # C. 处理前缀剥离 (提取 Backbone)
        # 如果键以 "backbone." 开头 -> 剥离前缀，变成 "esm_model..."
        # 如果键不以 "backbone." 开头 -> 保持原样 (兼容原始 ResidueEncoderBeta 权重)
        if prefix_to_strip and k.startswith(prefix_to_strip):
            k = k[len(prefix_to_strip):]
        
        clean_state_dict[k] = v

    return clean_state_dict



def check_rxnfp(embs, dim, eps_norm=1e-6, eps_vec_std=1e-6, max_abs_thr=1e3):
    """
    严格检查 RXNFP 生成的嵌入向量质量，确保不放过坏数据，也不误杀重复的好数据。
    
    Args:
        embs: List or np.ndarray, shape [B, D]
        dim: 预期维度
        eps_norm: L2范数下限，低于此值视为全零/无效向量
        eps_vec_std: 单向量内部的标准差下限，低于此值视为常数向量（如全0.5）
        max_abs_thr: 数值爆炸阈值
    """
    if embs is None:
        raise RuntimeError("[RXNFP] convert_batch returned None.")
    if len(embs) == 0:
        raise RuntimeError("[RXNFP] convert_batch returned empty embeddings.")

    # 1. 基础形状与数值类型检查
    X = np.asarray(embs, dtype=np.float32)
    if X.ndim != 2:
        raise RuntimeError(f"[RXNFP] embeddings must be 2D [B,D], got shape={X.shape}")
    if X.shape[1] != dim:
        raise RuntimeError(f"[RXNFP] dim mismatch: got D={X.shape[1]} vs expected_dim={dim}")

    if not np.isfinite(X).all():
        raise RuntimeError("[RXNFP] nan/inf detected in embeddings")

    # 2. 幅值粗检 (全局)
    max_abs = float(np.max(np.abs(X)))
    if max_abs > max_abs_thr:
        raise RuntimeError(f"[RXNFP] abnormal magnitude: max|x|={max_abs:.3e} (> {max_abs_thr})")

    # 3. 逐样本严格检查 (Instance-wise Check)
    # 不使用 Batch 统计量（如 mean_std 或 cos），因为 batch 内存在重复或相似样本是合法的。
    # 我们只关心：每个向量自己是否是个“好”向量。
    
    # 3.1 检查是否为零向量 (L2 Norm)
    norms = np.linalg.norm(X, axis=1)
    bad_norm_indices = np.where(norms <= eps_norm)[0]
    if len(bad_norm_indices) > 0:
        idx = int(bad_norm_indices[0])
        raise RuntimeError(f"[RXNFP] near-zero embedding detected at index {idx}: norm={float(norms[idx]):.3e}")

    # 3.2 检查是否为退化/常数向量 (Vector Std)
    # 计算每个向量内部维度的标准差。如果 std 极小，说明该向量所有维度值几乎相同（例如全为0.1）。
    # 对于 BERT 类模型，输出向量通常包含丰富信息，不应是平坦的。
    vec_stds = np.std(X, axis=1)
    bad_std_indices = np.where(vec_stds <= eps_vec_std)[0]
    if len(bad_std_indices) > 0:
        idx = int(bad_std_indices[0])
        val_sample = X[idx][:5] # 打印前几个值用于调试
        raise RuntimeError(f"[RXNFP] degenerate (constant) embedding detected at index {idx}: "
                           f"vec_std={float(vec_stds[idx]):.3e} (<= {eps_vec_std}). "
                           f"First 5 vals: {val_sample}")

    # Pass: 所有样本均通过独立性检查
    return



def format_prediction_to_string(pred_tensor):
    """
    Format predictions to string representations (e.g., [[185], [189]] and [1, 1]).
    Args:
        pred_tensor: shape (seq_len,), containing class indices.
    Returns:
        site_labels_str: e.g. "[[185], [189]]" (1-based indices)
        site_types_str: e.g. "[1, 1]"
    """
    indices = []
    types = []
    
    # pred_tensor is assumed to be of valid sequence length (no padding)
    for idx, p in enumerate(pred_tensor):
        p_val = p.item()
        if p_val != 0: # 0 is background
            indices.append([idx + 1]) # 1-based index
            types.append(int(p_val))
            
    return str(indices), str(types)
# =========================================== Bottom@Utilities Functions =================================================== 





# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
def compute_binary_classification_metrics(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc', 'ap', 'mcc']
) -> Dict[str, List[float]]:
    """
    Compute binaryclassification metrics.
    Args:
        predictions: Predicted labels [batch_size, seq_len] or [num_samples]，Binary predictions (0/1) of shape ``[B, L]`` or ``[N]``.
        probabilities: Prediction probabilities [batch_size, seq_len] or [num_samples]，Positive‑class probabilities – only needed for AUC / AP.
        labels: True labels [batch_size, seq_len] or [num_samples]，Ground‑truth binary labels, same shape as *predictions*.
        valid_mask: Mask indicating valid positions [batch_size, seq_len] or [num_samples]
        metrics: List of metrics to compute
    Returns:
        字典results，每个指标是一个列表，列表中的每个元素对应一个样本的指标值
    """
    # Initialize results dictionary with empty lists for each metric
    results = {m: [] for m in metrics}
    
    # Check if inputs are batched
    is_batched = len(predictions.shape) > 1
    
    if is_batched:
        # Process each sample in the batch separately
        batch_size = predictions.shape[0]
        
        for i in range(batch_size):
            # Extract data for this sample
            sample_pred = predictions[i]
            sample_labels = labels[i]
            sample_mask = valid_mask[i] if valid_mask is not None else None
            sample_probs = probabilities[i] if probabilities is not None else None
            
            # Calculate metrics for this sample
            sample_metrics = compute_binary_classification_metrics_per_sample(
                sample_pred, sample_probs, sample_labels, sample_mask, metrics
            )
            
            # Add results to the lists
            for m in metrics:
                results[m].append(sample_metrics[m])
    else:
        # Single sample case
        sample_metrics = compute_binary_classification_metrics_per_sample(
            predictions, probabilities, labels, valid_mask, metrics
        )
        for m in metrics:
            results[m].append(sample_metrics[m])
    
    return results



def compute_binary_classification_metrics_per_sample(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    metrics: List[str]
) -> Dict[str, float]:
    if valid_mask is not None:
        mask = valid_mask.to(torch.bool)
        preds_flat = predictions[mask].view(-1)
        labels_flat = labels[mask].view(-1)
        probs_flat = probabilities[mask].view(-1) if probabilities is not None else None
    else:
        preds_flat = predictions.view(-1)
        labels_flat = labels.view(-1)
        probs_flat = probabilities.view(-1) if probabilities is not None else None

    if preds_flat.numel() == 0:
        return {m: 0.0 for m in metrics}

    preds_bin = preds_flat.long()
    labels_bin = labels_flat.long()

    TP = int(((preds_bin == 1) & (labels_bin == 1)).sum())
    FP = int(((preds_bin == 1) & (labels_bin == 0)).sum())
    TN = int(((preds_bin == 0) & (labels_bin == 0)).sum())
    FN = int(((preds_bin == 0) & (labels_bin == 1)).sum())
    total = TP + FP + TN + FN
    results = {}

    def div(numer: float, denom: float, zero_default: float = 0.0) -> float:
        return numer / denom if denom != 0 else zero_default


    if "accuracy" in metrics:
        results["accuracy"] = div(TP + TN, total)
    if "precision" in metrics:
        results["precision"] = div(TP, TP + FP)
    if "recall" in metrics:
        # Also the *overlap_score*
        results["recall"] = div(TP, TP + FN)
    if "specificity" in metrics:
        # If denominator 0 (TN+FP==0), follow original: specificity = 1
        results["specificity"] = div(TN, TN + FP, zero_default=1.0)
    if "fpr" in metrics:
        results["fpr"] = div(FP, FP + TN)
    if "f1" in metrics:
        prec = results.get("precision", div(TP, TP + FP))
        rec = results.get("recall", div(TP, TP + FN))
        results["f1"] = div(2 * prec * rec, prec + rec)
    if "mcc" in metrics:
        denom = torch.sqrt(torch.tensor((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), dtype=torch.float64))
        results["mcc"] = div((TP * TN) - (FP * FN), denom.item())

    if probs_flat is not None and ("auc" in metrics or "ap" in metrics):
        # Convert only once to NumPy for sklearn compatibility
        labels_np = labels_bin.cpu().numpy()
        probs_np = probs_flat.cpu().numpy()

        if "auc" in metrics:
            try:
                results["auc"] = roc_auc_score(labels_np, probs_np)
            except ValueError:
                results["auc"] = 0.0
        if "ap" in metrics:
            try:
                results["ap"] = average_precision_score(labels_np, probs_np)
            except ValueError:
                results["ap"] = 0.0
    else:
        if "auc" in metrics:
            results["auc"] = 0.0
        if "ap" in metrics:
            results["ap"] = 0.0

    for m in metrics:
        results.setdefault(m, 0.0)

    return results
    


def compute_multiple_classification_metrics(
    pred: Union[torch.Tensor, List],
    gt: Union[torch.Tensor, List],
    num_site_types: int,
    valid_mask: Optional[torch.Tensor] = None
) -> Dict[str, List[float]]:
    """
    计算多分类指标，每个样本单独计算。
    
    Args:
        pred: 预测类别，形状为[batch_size, seq_len]或[num_samples]
        gt: 真实类别，形状为[batch_size, seq_len]或[num_samples]
        num_site_types: 类别数量，包括背景类
        valid_mask: 掩码，指示哪些位置是有效的，形状为[batch_size, seq_len]或[num_samples]
    Returns:
        字典metrics，每个指标是一个列表，列表中的每个元素对应一个样本的指标值
    """

    metrics = defaultdict(list)
    
    is_batched = False
    if isinstance(pred, torch.Tensor):
        is_batched = len(pred.shape) > 1
    elif isinstance(pred, list) and isinstance(pred[0], list):
        is_batched = True
    
    if is_batched:
        batch_size = len(pred) if isinstance(pred, list) else pred.shape[0]
        for i in range(batch_size):
            if isinstance(pred, torch.Tensor):
                sample_pred = pred[i]
                sample_gt = gt[i]
                sample_mask = valid_mask[i] if valid_mask is not None else None
            else:
                sample_pred = pred[i]
                sample_gt = gt[i]
                sample_mask = valid_mask[i] if valid_mask is not None else None
            
            # 计算单个样本的指标
            sample_metrics = compute_multiple_classification_metrics_per_sample(
                sample_pred, sample_gt, num_site_types, sample_mask
            )
            
            for metric_name, values in sample_metrics.items():
                metrics[metric_name].extend(values)
    else:
        # 单个样本情况
        sample_metrics = compute_multiple_classification_metrics_per_sample(
            pred, gt, num_site_types, valid_mask
        )
        
        for metric_name, values in sample_metrics.items():
            metrics[metric_name].extend(values)
    
    return metrics



def compute_multiple_classification_metrics_per_sample(
    pred: Union[torch.Tensor, List],
    gt: Union[torch.Tensor, List],
    num_site_types: int,
    valid_mask: Optional[torch.Tensor] = None
) -> Dict[str, List[float]]:
    """
    为单个样本计算多分类指标。
    
    Args:
        pred: 预测类别，一维列表或张量
        gt: 真实类别，一维列表或张量
        num_site_types: 类别数量，包括背景类
        valid_mask: 掩码，指示哪些位置是有效的
        
    Returns:
        包含各种指标的字典，每个指标是一个单元素列表
    """

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu()
        if valid_mask is not None:
            pred = pred[valid_mask.detach().cpu().bool()]
        pred = pred.tolist()
    elif valid_mask is not None and isinstance(valid_mask, torch.Tensor):
        # 如果pred已经是列表但有掩码
        valid_indices = valid_mask.nonzero().squeeze(-1).tolist()
        pred = [pred[i] for i in valid_indices]
    
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu()
        if valid_mask is not None:
            gt = gt[valid_mask.detach().cpu().bool()]
        gt = gt.tolist()
    elif valid_mask is not None and isinstance(valid_mask, torch.Tensor):
        # 如果gt已经是列表但有掩码
        valid_indices = valid_mask.nonzero().squeeze(-1).tolist()
        gt = [gt[i] for i in valid_indices]
    
    metrics = defaultdict(list)
    
    # 如果没有有效预测，返回空结果
    if not pred or not gt:
        return metrics
    
    metrics['multi_class_mcc'].append(matthews_corrcoef(gt, pred))
    recall_list = recall_score(gt, pred, average=None, labels=range(num_site_types), zero_division=0)
    cm = multilabel_confusion_matrix(gt, pred, labels=range(num_site_types))
    precision_list = precision_score(gt, pred, average=None, labels=range(num_site_types), zero_division=0)
    f1_list = f1_score(gt, pred, average=None, labels=range(num_site_types), zero_division=0)

    for class_idx in range(num_site_types):
        metrics[f"type_{class_idx}_recall"].append(recall_list[class_idx])
        metrics[f"type_{class_idx}_precision"].append(precision_list[class_idx])
        metrics[f"type_{class_idx}_f1"].append(f1_list[class_idx])

        TN, FP, FN, TP = cm[class_idx].ravel()
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        metrics[f"type_{class_idx}_fpr"].append(fpr)
    
    return metrics



class Predictior(nn.Module):
    """
    EasIFA的预测网络
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(Predictior, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 2, out_dim),
        )
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.net(x)
    


def SupConHardLoss(model_emb, temp, n_pos):
    '''
    return the SupCon-Hard loss
    features:  
        model output embedding, dimension [bsz, n_all, out_dim], 
        where bsz is batchsize, 
        n_all is anchor, pos, neg (n_all = 1 + n_pos + n_neg)
        and out_dim is embedding dimension
    temp:
        temperature     
    n_pos:
        number of positive examples per anchor
    '''
    # l2 normalize every embedding
        # F.normalize():
    #   - dim=-1: 在最后一个维度（out_dim）上归一化
    #   - p=2: 使用 L2 范数（欧式距离）
    #
    # 数学公式: features[i, j] = model_emb[i, j] / ||model_emb[i, j]||_2
    #
    # 输入: model_emb.shape = (6000, 40, 256)
    # 输出: features.shape = (6000, 40, 256)
    #       每个向量的 L2 范数都为 1
    features = F.normalize(model_emb, dim=-1, p=2)


    # features_T is [bsz, outdim, n_all], for performing batch dot product
    # 输入: features.shape = (bsz, n_all, out_dim) = (6000, 40, 256)
    # 输出: features_T.shape = (bsz, out_dim, n_all) = (6000, 256, 40)
    features_T = torch.transpose(features, 1, 2)


    # anchor is the first embedding 
    anchor = features[:, 0]


    # anchor is the first embedding 
    # 步骤拆解:
    #
    # 1. anchor.unsqueeze(1): 增加维度
    #    输入: anchor.shape = (6000, 256)
    #    输出: anchor.unsqueeze(1).shape = (6000, 1, 256)
    #
    # 2. torch.bmm(): 批量矩阵乘法 (batch matrix multiplication)
    #    矩阵 A: (6000, 1, 256)  ← anchor
    #    矩阵 B: (6000, 256, 40) ← features_T
    #    结果 C: (6000, 1, 40)
    #
    #    计算: C[i, 0, j] = A[i, 0, :] · B[i, :, j]
    #                     = anchor[i] · features[i, j]
    #                     = 两个 L2 归一化向量的点积
    #                     = 余弦相似度
    #
    # 3. /temp: 温度缩放
    #    temp = 0.1 → 相似度放大 10 倍
    #    作用: 使分布更锐利，增大正负样本的差距
    #
    # 输出: anchor_dot_features.shape = (6000, 1, 40)
    # 结果示例 (某个样本):
    anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T)/temp 


    # anchor_dot_features now [bsz, n_all], contains 
    # anchor_dot_features[0] = [10.0, 8.5, 9.2, ..., 2.3, 1.8, ...]
    #                           ^     ^~~正样本~^      ^~~负样本~~^
    #                           自己
    anchor_dot_features = anchor_dot_features.squeeze(1)


    # deduct by max logits, which will be 1/temp since features are L2 normalized 
    # 数值稳定性技巧 (log-sum-exp trick):
    #
    # 原因: anchor 与自己的点积 = 1 (L2 归一化后)
    #       经过温度缩放: 1/0.1 = 10
    #       这是所有 logit 中的最大值
    #
    # 减去最大值的好处:
    # 1. 防止 exp(大数) 溢出
    # 2. 不改变 softmax 结果: softmax(x - c) = softmax(x)
    #
    # 示例:
    # anchor_dot_features[0] = [10.0, 8.5, 9.2, ..., 2.3]
    # logits[0] = [0.0, -1.5, -0.8, ..., -7.7]
    #             ^
    #             anchor 与自己的相似度变为 0
    logits = anchor_dot_features - 1/temp


    # the exp(z_i dot z_a) excludes the dot product between itself
    # exp_logits is of size [bsz, n_pos+n_neg]
    # logits[:, 1:]: 跳过第 0 列（anchor 与自己）
    #
    # 输入: logits.shape = (6000, 40)
    # 输出: exp_logits.shape = (6000, 39)  # 包含 9 个正样本 + 30 个负样本
    exp_logits = torch.exp(logits[:, 1:])


    # 1. exp_logits.sum(1): 沿第 1 维求和
    #    输入: exp_logits.shape = (6000, 39)
    #    输出: sum.shape = (6000,)
    #    计算: sum[i] = Σ_{j=1}^{39} exp(logits[i, j])
    #
    # 2. torch.log(): 取对数
    #    log_sum[i] = log(Σ_{j=1}^{39} exp(logits[i, j]))
    #
    # 3. n_pos * ...: 乘以正样本数量 (9)
    #    作用: 后续会除以 n_pos，这里先乘以 n_pos 是为了数学上的对称性
    #
    # 输出: exp_logits_sum.shape = (6000,)
    exp_logits_sum = n_pos * torch.log(exp_logits.sum(1)) # size [bsz], scale by n_pos


    # 输入: logits.shape = (6000, 40)
    # 切片: logits[:, 1:10].shape = (6000, 9)
    # 求和: pos_logits_sum.shape = (6000,)
    pos_logits_sum = logits[:, 1:n_pos+1].sum(1) #sum over all (anchor dot pos)
    log_prob = (pos_logits_sum - exp_logits_sum)/n_pos

    # - log_prob: 取负（因为我们要最大化 log 概率，等价于最小化负 log 概率）
    # .mean(): 对整个 batch 取平均
    #
    # 输出: loss.shape = ()  # 标量
    #
    # 解释:
    # - log_prob 越大 → loss 越小 → 模型越好
    # - log_prob 大 意味着:
    #   1. pos_logits_sum 大：anchor 与正样本相似度高
    #   2. exp_logits_sum 小：anchor 与负样本相似度低
    loss = - log_prob.mean()
    return loss  



def encode_residue_feature(
    flat_features: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    对一个大的扁平特征张量进行编码。

    Args:
        flat_features (torch.Tensor): 位于CPU的完整特征张量。
        model (torch.nn.Module): 用于编码的神经网络模型。
        device (torch.device): 计算设备 (e.g., torch.device("cuda:0"))。
        batch_size (int): 每个批次的大小。
        use_amp (bool): 是否启用自动混合精度 (AMP)。
        dtype (torch.dtype): 混合精度使用的数据类型 (e.g., torch.float16)。

    Returns:
        torch.Tensor: 位于CPU的完整编码后张量。
    """
    model.eval()
    model.to(device)
    encoded_chunks = []

    with torch.inference_mode(), autocast(enabled=use_amp, dtype=dtype):
        for i in range(0, flat_features.shape[0], batch_size):
            chunk_gpu = flat_features[i : i + batch_size].to(device)
            encoded_chunk = model(chunk_gpu)
            encoded_chunks.append(encoded_chunk.cpu())
            
    return torch.cat(encoded_chunks, dim=0)



def create_defaultdict():
    return defaultdict(list)



def parse_embedding_dict(embedding_dict: Dict[str, torch.Tensor]) -> Set[str]:
    """
    从 embedding_dict 中选出“形状为 [L, D] 且 L>0”的合法 uid 集合。
    """
    valid_uids = set()
    for uid, emb in embedding_dict.items():
        if isinstance(emb, torch.Tensor) and emb.ndim == 2 and emb.shape[0] > 0:
            valid_uids.add(str(uid).strip())
    return valid_uids



def process_row(row, cluster_mode: str, ec_level: int):
    """
    只负责根据“干净的 row ”构建临时数据池，不负责过滤数据：
        temp_cluster_site_pointers: cluster ID -> 位点类型 -> 残基指针列表

    要求 row 中已经存在：
        - 'Uniprot ID'     : 标准化后的字符串
        - 簇标签，根据 cluster_mode 决定下面两种的一种：
            * 'rxn_cluster_label'：当 cluster_mode == 'rxn_cluster'
            - 'valid_ec_list'  : 当 cluster_mode == 'ec', List[str]，由 parse_ec_list 得到的有效 EC 列表
        - 'Sequence_length': 与 embedding_dict 对应的长度
        - 'labels'         : np.ndarray[seq_len]，由 parse_site_labels 得到的有效标签向量
    
    Args:
        row: DataFrame 的一行
        cluster_mode: "ec" 或 "rxn_cluster"
        ec_level: EC 截断层级 (仅当 cluster_mode="ec" 时有效)
    Returns:
        temp_cluster_site_pointers (process_row的返回值之一) = {
            # 以4级EC为例
            '1.1.1.1': {
                0: [('P12345', 0), ('P12345', 1), ..., ('P12345', 15), ('P12345', 17), ...],  # 非活性位点的指针
                1: [('P12345', 16), ('P12345', 22)],                                            # 2个结合位点的指针
                2: [('P12345', 73)]                                                             # 1个催化位点的指针
            }
        }
            其中residue_index 是 0-based 索引
    """
    uniprot_id = str(row['Uniprot ID']).strip()
    labels: np.ndarray = row['labels']            # np.ndarray[seq_len]
    seq_len = int(row['Sequence_length'])
    assert labels.shape[0] == seq_len, f"labels length {labels.shape[0]} != seq_len {seq_len} for {uniprot_id}"

    raw_clusters = []
    if cluster_mode == "ec":
        valid_ec_list = row['valid_ec_list']
        for ec in valid_ec_list:
            ec = str(ec).strip()
            if ec_level < 4:
                raw_clusters.append('.'.join(ec.split('.')[:ec_level]))
            else:
                raw_clusters.append(ec)     
    elif cluster_mode == "rxn_cluster":
        raw_clusters.append(str(row['rxn_cluster_label']))
    unique_clusters = list(set(raw_clusters))   # 去重，例如 1.1.1.1 和 1.1.1.2 在 level 3 下归为同一个 Cluster 1.1.1

    # Cluster 内位点指针
    # temp_cluster_site_pointers: ClusterID -> SiteType -> List[(uid, res_idx)]
    temp_cluster_site_pointers = defaultdict(create_defaultdict)
    for cluster in unique_clusters:
        for i in range(seq_len):
            site_type = labels[i]
            temp_cluster_site_pointers[cluster][site_type].append((uniprot_id, i))
            
    return (uniprot_id, unique_clusters, temp_cluster_site_pointers, labels)



def build_data_pools(
    dataframe: pd.DataFrame, 
    embedding_dict: dict, 
    num_workers: int, 
    ec_level: int = 4,
    cluster_mode: str = "ec"
):
    """
    使用统一的dataframe，构建 
        * Cluster ->蛋白质ID， 
        * 蛋白质ID ->Clusters,
        * 蛋白质ID ->标签张量,
        * Cluster ->位点类型->残基指针

    Args:
        dataframe (pd.DataFrame): 来自 EnzymeActiveSiteDataset3 的 dataset_df，该dataset_df是所有操作的数据基础。
        embedding_dict (dict): 用于此数据集的残基嵌入字典。embedding_dict有两个作用：
            * 过滤 dataframe 中没有嵌入特征的蛋白质ID。虽然dataframe是统一的、经过多重过滤的数据框，但是在数据集类的__getitem__函数以及模型的forward函数中，仍然可能出现处理失败的情况，这是不可避免的。embedding_dict的作用就是利用embedding_dict.items()中的key，过滤掉dataframe中没有嵌入特征的蛋白质ID。
            * 提供残基特征，用于后续构建平滑的特征张量 flat_features。
        num_workers (int): 用于并行处理的核数。
        ec_level (int): The level of EC number to use for grouping (e.g., 3 or 4). Default is 4.  (仅 cluster_mode="ec" 时生效)
        cluster_mode: "ec" (默认) 或 "rxn_cluster"，决定是按照 EC 号还是反应簇进行分组。


    Returns:
        tuple:(以cluster_mode="ec"且ec_level=4为例)
            - id_ec_map (dict): UniProt ID 到其 EC 号列表的映射。
                id_ec_map = {
                    'P19367': ['2.7.1.1'],                  # 只有一个 EC
                    'P52789': ['2.7.1.1', '3.4.21.4'],      # 多个 EC
                }
            - ec_id_set: Dict[str, Set[str]]
                ec_id_set = {
                    '2.7.1.1': {'P19367', 'P52789', ...},
                    '3.4.21.4': {'P52789', ...},
                }
            - ec_site_pools (dict): EC 号到其包含的位点池的映射。
                结构: {'EC号': {位点类型: [(UniProt ID, 0-based残基索引), ...]}}
                示例:
                {
                    '2.7.1.1': { # 己糖激酶的EC号    
                        0: [('P19367', 0), ('P19367', 1), ..., ('P52789', 0), ...],       # 非活性位点 (类型 0) 的指针列表
                        1: [('P19367', 152), ('P19367', 205), ('P52789', 150)],           # 结合位点 (类型 1) 的指针列表,P19367的第153、206号残基和P52789的第151号残基是结合位点   
                        2: [('P19367', 177)]                                              # 催化位点 (类型 2) 的指针列表,P19367的第178号残基是催化位点
                        3: [('P19367', 50), ('P52789', 75), ...]                          # other位点 (类型 3) 的指针列表
                    },
                    '3.4.21.4': {
                        # ... 胰蛋白酶的位点指针
                    }
                }
            - labels_dict: Dict[str, torch.Tensor], 0-based
                labels_dict['P19367'] = tensor(
                    [0, 0, 0, ..., 1, 0, 2, ...], dtype=torch.int64
                )
            - flat_features (torch.Tensor): 对所有蛋白质残基特征按 uids_ordered 顺序拼接得到的统一特征矩阵。
                假设第 k 个蛋白 uid_k 的残基数为 L_k，总残基数 N = sum_k L_k，特征维度为 D，则:
                    flat_features.shape == [N, D]
                其中 flat_features[uid_to_offset[uid] + i] 对应蛋白 uid 的第 i 个残基 (0-based) 的特征向量。
            - ec_site_pools_indexed (dict): EC 号到“位点类型 -> 整数索引张量”的映射，索引是 flat_features 的 0-based 行号。
                结构: {'EC号': {位点类型: tensor([idx_0, idx_1, ...], dtype=torch.long)}}
                示例:
                {
                    '2.7.1.1': {
                        0: tensor([   0,    1,    2,  ...,  500], dtype=torch.long),  # 类型 0 残基在 flat_features 中的行索引
                        1: tensor([ 152,  205,  350,  ...], dtype=torch.long),        # 类型 1 残基在 flat_features 中的行索引
                        2: tensor([ 178,  ...], dtype=torch.long),
                        3: tensor([  50,   75, ...], dtype=torch.long),
                    },
                    '3.4.21.4': {
                        # ...
                    }
                }
            - uids_ordered (List[str]): 按字典序排序后的、最终被保留的 UniProt ID 列表。
                该列表决定 flat_features 的拼接顺序，也是 uid_to_offset 构造的依据：
                    第 k 个 uid = uids_ordered[k] 的残基特征依次出现在 flat_features
                    [uid_to_offset[uid] : uid_to_offset[uid] + L_uid] 这一段。
            - uid_to_offset (Dict[str, int]): UniProt ID 到其在 flat_features 中起始行号 (0-based) 的映射。
                示例:
                    uid_to_offset = {
                        'P19367': 0,        # P19367 的第 0 个残基对应 flat_features[0]
                        'P52789': 350,      # P52789 的第 0 个残基对应 flat_features[350]
                        # ...
                    }
                对于任意 uid 与残基索引 res_idx (0-based)，其在 flat_features 中的全局索引为:
                    global_idx = uid_to_offset[uid] + res_idx
    """
    print(f"Building unified data pools from DataFrame (size: {len(dataframe)})...")

    # 在模型处理dataframe的某些样本时，可能会出现问题，导致embedding_dict.keys()可能是dataframe['Uniprot ID']的子集
    valid_embedding_uids = parse_embedding_dict(embedding_dict)
    df = dataframe.copy()   # 不要直接修改传入的 dataframe，copy 一份
    df = df[df['Uniprot ID'].isin(valid_embedding_uids)]
    print(f"[build_data_pools] Filtered by embeddings: {len(dataframe)} -> {len(df)}")


    # 1 收集基础信息和指针池
    worker_func = partial(process_row, cluster_mode=cluster_mode, ec_level=ec_level)
    if num_workers > 0:
        pandarallel.initialize(nb_workers=num_workers, progress_bar=False, verbose=0)
        results = df.parallel_apply(worker_func, axis=1)
    else:
        tqdm.pandas(desc="Processing rows")
        results = df.progress_apply(worker_func, axis=1)

    id_cluster_map = {}
    cluster_id_set = defaultdict(set)
    labels_dict = {}
    cluster_site_pools = defaultdict(create_defaultdict)
    for result in tqdm(results, desc="Merging results"):
        uniprot_id, unique_clusters, temp_cluster_site_pools, labels_array = result
        
        id_cluster_map[uniprot_id] = unique_clusters    # 记录 ID -> Clusters 映射
        
        for cluster in unique_clusters:                 # 记录 Cluster -> IDs 集合
            cluster_id_set[cluster].add(uniprot_id)
        
        labels_dict[uniprot_id] = torch.from_numpy(labels_array)    # 记录 Labels
        
        for cluster, type_pools in temp_cluster_site_pools.items(): # 合并位点指针 (Cluster -> Type -> Pointers)
            for site_type, pointers in type_pools.items():
                cluster_site_pools[cluster][site_type].extend(pointers)
    

    # 2 构建对齐的 flat_features 张量
    final_valid_uids = set(id_cluster_map.keys())
    uids_ordered = sorted(list(final_valid_uids))
    
    uid_to_offset = {}
    flat_features_list = []
    current_offset = 0
    for uid in uids_ordered:
        tensor = embedding_dict[uid]
        uid_to_offset[uid] = current_offset
        flat_features_list.append(tensor)
        current_offset += tensor.shape[0]
    flat_features = torch.cat(flat_features_list, dim=0)


    # 3 将指针池转换为整数索引池
    cluster_site_pools_indexed = defaultdict(dict)
    for cluster, type_pools in cluster_site_pools.items():
        for site_type, pointers in type_pools.items():
            indices = [uid_to_offset[uid] + res_idx for uid, res_idx in pointers if uid in uid_to_offset]
            if indices:
                cluster_site_pools_indexed[cluster][site_type] = torch.tensor(indices, dtype=torch.long)
    
    print(f"build_data_pools completed, processed {len(id_cluster_map)} proteins.")
    return (id_cluster_map, cluster_id_set, cluster_site_pools, labels_dict, flat_features, cluster_site_pools_indexed, uids_ordered, uid_to_offset)



@torch.no_grad()
def cluster_site_type_prediction(
    site_centers: Dict[str, Dict[int, torch.Tensor]],
    site_stats: Dict[str, Dict[int, Dict[str, float]]],
    query_residue_features: torch.Tensor,
    id_cluster_query: Dict[str, List[str]],
    labels_dict_query: Dict[str, torch.Tensor],
    uid_to_offset_query: Dict[str, int],
    device: torch.device,
    prob_mode: str = "distance",
    use_priors: bool = True,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[Dict[str, List[float]], Dict[str, torch.Tensor]]:
    """
    Unified evaluation function for active site type prediction using cluster centers.

    基于给定的簇中心（site_centers）对查询集进行位点类型预测和评估。

    Args:
        * site_centers 结构为 {'cluster_id': {位点类型: 质心张量}}
            {
                'cluster_id_0': {
                    0: tensor([0.12, -0.34, ..., 0.56]),  # 0号簇下所有参考集非活性位点(类型0)的平均嵌入向量，形状: [output_dim], e.g., [128]
                    1: tensor([0.88,  0.15, ..., -0.21]), # 0号簇下所有参考集结合位点(类型1)的平均嵌入向量，形状: [output_dim], e.g., [128]
                    2: tensor([0.95, -0.05, ..., 0.08]),  # 0号簇下所有参考集催化位点(类型2)的平均嵌入向量，形状: [output_dim], e.g., [128]
                    3: tensor([-0.95, 0.05, ..., 0.08])   # 0号簇下所有参考集other位点(类型3)的平均嵌入向量，形状: [output_dim], e.g., [128]
                },
                'cluster_id_1': {
                    0: tensor([...]),
                    1: tensor([...]),
                    2: tensor([...]),
                    3: tensor([...])
                },
                # ... 其他cluster_id号
            }
        * site_stats   结构为 {'cluster_id': {位点类型: {'sigma2': float, 'count': int}}}
        * query_residue_features (torch.Tensor): 需要评估的查询集扁平残基嵌入矩阵，设查询集中共有 N_query 个残基、每个残基嵌入维度为 D，则：query_residue_features.shape == [N_query, D]
            对于某个查询蛋白 uid，其第 `res_idx` 个残基（0-based）在该矩阵中的全局行号为：global_idx = uid_to_offset_query[uid] + res_idx
        * id_cluster_query (Dict[str, List[str]]): 查询集“UniProt ID -> clusters”的映射，通常由
            `build_data_pools(query_data_path, ...)` 返回的 `id_cluster_map` 获得。
            示例:
                id_cluster_query = {
                    'P19367': ['cluster_id_0'],
                    'P52789': ['cluster_id_0', 'cluster_id_1'],
                    # ...
                }
        * labels_dict_query (Dict[str, torch.Tensor]): 查询集“UniProt ID -> 残基标签向量”的映射，
            示例:
                labels_dict_query['P19367'] = tensor([0, 0, 0, ..., 1, 0, 2, ...], dtype=torch.int64)
            其中取值含义：0=非活性，1=结合位点，2=催化位点，3=other 位点。

        * uid_to_offset_query (Dict[str, int]): 查询集“UniProt ID -> 在 query_residue_features 中起始行号(0-based)”的映射，
            示例:
                uid_to_offset_query = {
                    'P19367': 0,      # P19367 的第 0 个残基对应 query_residue_features[0]
                    'P52789': 350,    # P52789 的第 0 个残基对应 query_residue_features[350]
                    # ...
                }
            因此对于任意 uid 及其第 res_idx 个残基（0-based），其在 query_residue_features 中的行索引为: global_idx = uid_to_offset_query[uid] + res_idx
        * prob_mode:
            - "gaussian": 使用高斯判别形式的距离 -> logits
            - "distance": 使用与质心的距离直接构造 logits
        * use_priors: 仅在 "gaussian" 模式下是否使用类别先验 π_j。
        * temperature: logits 除以该温度后再 softmax，T>1 概率更平滑。

    Returns:
        * all_metrics：一个字典。
            all_metrics = {
                # 总体指标：多分类MCC（每个样本一个分数）
                'multi-class mcc':      [0.8123, 0.7450, 0.7904, ...],   # 每个样本的多分类MCC，列表长度等于测试集样本数。

                # 按类别统计：召回率（Recall = TP / (TP+FN)）
                'type_0_recall':        [0.985, 0.972, 0.990, ...],      # 类别0(非活性)在每个样本上的召回
                'type_1_recall':        [0.905, 0.880, 0.915, ...],      # 类别1(结合)在每个样本上的召回
                'type_2_recall':        [0.720, 0.680, 0.750, ...],      # 类别2(催化)在每个样本上的召回
                'type_3_recall':        [0.640, 0.610, 0.670, ...],      # 类别3(other)在每个样本上的召回

                # 按类别统计：精确率（Precision = TP / (TP+FP)）
                'type_0_precision':     [0.992, 0.988, 0.991, ...],      # 类别0在每个样本上的精确率
                'type_1_precision':     [0.865, 0.840, 0.872, ...],      # 类别1在每个样本上的精确率
                'type_2_precision':     [0.760, 0.700, 0.780, ...],      # 类别2在每个样本上的精确率
                'type_3_precision':     [0.600, 0.590, 0.620, ...],      # 类别3在每个样本上的精确率

                # 按类别统计：F1（F1 = 2 * P * R / (P + R)）
                'type_0_f1':            [0.988, 0.980, 0.990, ...],      # 类别0在每个样本上的F1
                'type_1_f1':            [0.885, 0.860, 0.893, ...],      # 类别1在每个样本上的F1
                'type_2_f1':            [0.739, 0.690, 0.765, ...],      # 类别2在每个样本上的F1
                'type_3_f1':            [0.619, 0.600, 0.644, ...],      # 类别3在每个样本上的F1

                # 按类别统计：假阳性率 FPR（FPR = FP / (FP + TN)）
                'type_0_fpr':           [0.010, 0.015, 0.008, ...],      # 类别0在每个样本上的FPR
                'type_1_fpr':           [0.012, 0.020, 0.018, ...],      # 类别1在每个样本上的FPR
                'type_2_fpr':           [0.005, 0.007, 0.006, ...],      # 类别2在每个样本上的FPR
                'type_3_fpr':           [0.018, 0.022, 0.019, ...],      # 类别3在每个样本上的FPR
            }
        * per_sample_probs: 每个样本每个残基的4类概率，用于堆叠泛化元学习器
            {
                uniprot_id: Tensor[seq_len, 4]  # 概率顺序为类别 0,1,2,3
            }
    """
    num_classes = 4
    all_metrics = defaultdict(list)
    per_sample_probs: Dict[str, torch.Tensor] = {}
    query_uids = sorted(id_cluster_query.keys()) # 保证评估顺序一致性
    for uniprot_id in tqdm(query_uids, desc="Evaluating Query Set", disable=True):
        cluster_list = id_cluster_query[uniprot_id]
        if not cluster_list: continue
        cluster_id = cluster_list[0]     # $$只取第一个Cluster ID进行评估

        if cluster_id not in site_centers: continue
        if uniprot_id not in labels_dict_query or uniprot_id not in uid_to_offset_query: continue

        offset = uid_to_offset_query[uniprot_id]
        true_labels = labels_dict_query[uniprot_id]
        seq_len = true_labels.shape[0]
        residue_embs = query_residue_features[offset : offset + seq_len].to(device)

        centers = site_centers[cluster_id]
        if not centers: continue
        center_types = sorted(centers.keys())                                          # $$还需要考虑一个边界情况：某些cluster可能缺失某些位点类型的质心
        center_vectors = torch.stack([centers[t] for t in center_types]).to(device)    # center_vectors: [num_types, output_dim]，例如 [4, 128]
        
        # dist_matrix: 计算每个残基的输出嵌入与每个位点类型质心之间的欧氏距离
        #   - 含义: 距离矩阵，dist_matrix[i, j] 表示第 i 个残基与第 j 个类别质心的距离。
        #   - 形状: [seq_len, num_types]，例如 [450, 4] 
        dist_matrix = torch.cdist(residue_embs, center_vectors, p=2)
        sq_dists = dist_matrix ** 2

        # =========================================== Top@compute per_sample_probs ======================================================
        if prob_mode == "gaussian":
            sigma2_list = []
            count_list = []
            for t in center_types:
                st = site_stats[cluster_id][t] 
                sigma2_list.append(st["sigma2"])
                count_list.append(st["count"])
            sigma2 = torch.tensor(sigma2_list, device=device)          # [num_types]
            counts = torch.tensor(count_list, dtype=torch.float32, device=device)
            if use_priors:
                priors = counts / counts.sum()
                log_priors = torch.log(priors.clamp_min(1e-12))
            else:
                log_priors = torch.zeros_like(sigma2)
            logits_sub = -0.5 * sq_dists / sigma2.clamp_min(eps) + log_priors  # [seq_len, num_types]
        
        elif prob_mode == "distance":  # prob_mode == "distance"，直接用距离构造 logits（越近越大）
            logits_sub = -sq_dists  # [seq_len, num_types]
        
        if temperature is not None and temperature > 0: logits_sub = logits_sub / float(temperature)
        probs_sub = torch.softmax(logits_sub, dim=1)  # [seq_len, num_types]
        probs_full = torch.zeros(seq_len, num_classes, device=device, dtype=probs_sub.dtype)
        probs_full[:, center_types] = probs_sub
        per_sample_probs[uniprot_id] = probs_full.detach().to(torch.float32).cpu()
        # =========================================== Bottom@compute per_sample_probs =================================================== 

        
        # =========================================== Top@compute all_metrics ======================================================
        if prob_mode == "gaussian":
            preds = torch.argmax(probs_full, dim=1)
        elif prob_mode == "distance":
            # Distance模式: 严格沿用原代码逻辑，基于最小几何距离 (argmin)
            # 虽然数学上 argmin(dist) == argmax(-dist^2)，但这样写能最大程度保留原代码的逻辑
            # min_dist_indices: 对每个残基，找到距离最近的质心的索引
            #   - 含义: 一个向量，其第 i 个元素是第 i 个残基距离最近的质心的索引（0, 1, 2...）。
            #   - 形状: [seq_len]，例如 [450]
            min_dist_indices = torch.argmin(dist_matrix, dim=1)
            preds = torch.tensor([center_types[i] for i in min_dist_indices.cpu()], dtype=torch.long)

        # compute_multiple_classification_metrics 需要批次维度，因此使用 unsqueeze(0)
        sample_metrics = compute_multiple_classification_metrics(
            pred=preds.unsqueeze(0),
            gt=true_labels.unsqueeze(0),
            num_site_types=num_classes,
            valid_mask=torch.ones_like(true_labels, dtype=torch.bool).unsqueeze(0)
        )

        for metric_name, values in sample_metrics.items():
            if values:
                all_metrics[metric_name].extend(values)
        # =========================================== Bottom@compute all_metrics =================================================== 

    if not any(v for v in all_metrics.values()):
        print("Warning: No samples were evaluated. Check Cluster/EC overlap.")
        return {}, {} # Return empty dicts to match signature

    return all_metrics, per_sample_probs



@torch.no_grad()
def evaluate_site_type_prediction(
    reference_residue_features: torch.Tensor,
    query_residue_features: torch.Tensor,
    ec_site_pools_indexed_reference: Dict[str, Dict[int, torch.Tensor]],
    id_ec_query: Dict[str, List[str]],
    labels_dict_query: Dict[str, torch.Tensor],
    uid_to_offset_query: Dict[str, int],
    device: torch.device,
    prob_mode: str = "distance",
    use_priors: bool = True,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[Dict[str, List[float]], Dict[str, torch.Tensor]]:
    """
    * 当前evaluate_site_type_prediction函数在相同的输入（仅prob_mode参数不同）下，其预测的结果all_metrics也可能不一样。
    * 一定要确保reference_residue_features和query_residue_features是经过相同的模型或者一组模型处理得到的，否则评估结果没有意义。

    Args:
        * reference_residue_features (torch.Tensor): 用于构建簇中心的参考集扁平残基嵌入矩阵，设参考集中共有 N_ref 个残基、每个残基嵌入维度为 D，则：reference_residue_features.shape == [N_ref, D]
        * query_residue_features (torch.Tensor): 需要评估的查询集扁平残基嵌入矩阵，设查询集中共有 N_query 个残基、每个残基嵌入维度为 D，则：query_residue_features.shape == [N_query, D]
            对于某个查询蛋白 uid，其第 `res_idx` 个残基（0-based）在该矩阵中的全局行号为：global_idx = uid_to_offset_query[uid] + res_idx
        * ec_site_pools_indexed_reference (Dict[str, Dict[int, torch.Tensor]]):参考集的“EC 号 -> 位点类型 -> 残基全局索引集合”映射，
            结构:
                {
                    '2.7.1.1': {
                        0: tensor([   0,    1,  ...,  500], dtype=torch.long),  # 类型0残基在 reference_residue_features 中的行索引，0-based
                        1: tensor([ 152,  205,  ...],     dtype=torch.long),    # 类型1残基行索引
                        2: tensor([ 178,  ...],           dtype=torch.long),    # 类型2残基行索引
                        3: tensor([  50,   75, ...],      dtype=torch.long),    # 类型3残基行索引
                    },
                    '3.4.21.4': { ... },
                    # ...
                }
            这些索引用于从 reference_residue_features 中抽取所有属于某个 EC / 某个位点类型的残基嵌入，进而计算对应的簇中心（质心）与方差。
        * id_ec_query (Dict[str, List[str]]): 查询集“UniProt ID -> EC 号列表”的映射，通常由
            `build_data_pools(query_data_path, ...)` 返回的 `id_ec_map` 获得。
            示例:
                id_ec_query = {
                    'P19367': ['2.7.1.1'],
                    'P52789': ['2.7.1.1', '3.4.21.4'],
                    # ...
                }
        * labels_dict_query (Dict[str, torch.Tensor]): 查询集“UniProt ID -> 残基标签向量”的映射，
            示例:
                labels_dict_query['P19367'] = tensor([0, 0, 0, ..., 1, 0, 2, ...], dtype=torch.int64)
            其中取值含义：0=非活性，1=结合位点，2=催化位点，3=other 位点。

        * uid_to_offset_query (Dict[str, int]): 查询集“UniProt ID -> 在 query_residue_features 中起始行号(0-based)”的映射，
            示例:
                uid_to_offset_query = {
                    'P19367': 0,      # P19367 的第 0 个残基对应 query_residue_features[0]
                    'P52789': 350,    # P52789 的第 0 个残基对应 query_residue_features[350]
                    # ...
                }
            因此对于任意 uid 及其第 res_idx 个残基（0-based），其在 query_residue_features 中的行索引为: global_idx = uid_to_offset_query[uid] + res_idx
        * prob_mode:
            - "gaussian": 使用高斯判别形式的距离 -> logits
            - "distance": 使用与质心的距离直接构造 logits
        * use_priors: 仅在 "gaussian" 模式下是否使用类别先验 π_j。
        * temperature: logits 除以该温度后再 softmax，T>1 概率更平滑。
    Returns:
        * all_metrics：一个字典。
            all_metrics = {
                # 总体指标：多分类MCC（每个样本一个分数）
                'multi-class mcc':      [0.8123, 0.7450, 0.7904, ...],   # 每个样本的多分类MCC，列表长度等于测试集样本数。

                # 按类别统计：召回率（Recall = TP / (TP+FN)）
                'type_0_recall':        [0.985, 0.972, 0.990, ...],      # 类别0(非活性)在每个样本上的召回
                'type_1_recall':        [0.905, 0.880, 0.915, ...],      # 类别1(结合)在每个样本上的召回
                'type_2_recall':        [0.720, 0.680, 0.750, ...],      # 类别2(催化)在每个样本上的召回
                'type_3_recall':        [0.640, 0.610, 0.670, ...],      # 类别3(other)在每个样本上的召回

                # 按类别统计：精确率（Precision = TP / (TP+FP)）
                'type_0_precision':     [0.992, 0.988, 0.991, ...],      # 类别0在每个样本上的精确率
                'type_1_precision':     [0.865, 0.840, 0.872, ...],      # 类别1在每个样本上的精确率
                'type_2_precision':     [0.760, 0.700, 0.780, ...],      # 类别2在每个样本上的精确率
                'type_3_precision':     [0.600, 0.590, 0.620, ...],      # 类别3在每个样本上的精确率

                # 按类别统计：F1（F1 = 2 * P * R / (P + R)）
                'type_0_f1':            [0.988, 0.980, 0.990, ...],      # 类别0在每个样本上的F1
                'type_1_f1':            [0.885, 0.860, 0.893, ...],      # 类别1在每个样本上的F1
                'type_2_f1':            [0.739, 0.690, 0.765, ...],      # 类别2在每个样本上的F1
                'type_3_f1':            [0.619, 0.600, 0.644, ...],      # 类别3在每个样本上的F1

                # 按类别统计：假阳性率 FPR（FPR = FP / (FP + TN)）
                'type_0_fpr':           [0.010, 0.015, 0.008, ...],      # 类别0在每个样本上的FPR
                'type_1_fpr':           [0.012, 0.020, 0.018, ...],      # 类别1在每个样本上的FPR
                'type_2_fpr':           [0.005, 0.007, 0.006, ...],      # 类别2在每个样本上的FPR
                'type_3_fpr':           [0.018, 0.022, 0.019, ...],      # 类别3在每个样本上的FPR
            }
        * per_sample_probs: 每个样本每个残基的4类概率，用于堆叠泛化元学习器
            {
                uniprot_id: Tensor[seq_len, 4]  # 概率顺序为类别 0,1,2,3
            }
    """
    # 计算参考集的位点质心
    """
    * site_centers 结构为 {'cluster_id': {位点类型: 质心张量}}
        {
            'cluster_id_0': {
                0: tensor([0.12, -0.34, ..., 0.56]),  # 0号簇下所有参考集非活性位点(类型0)的平均嵌入向量，形状: [output_dim], e.g., [128]
                1: tensor([0.88,  0.15, ..., -0.21]), # 0号簇下所有参考集结合位点(类型1)的平均嵌入向量，形状: [output_dim], e.g., [128]
                2: tensor([0.95, -0.05, ..., 0.08]),  # 0号簇下所有参考集催化位点(类型2)的平均嵌入向量，形状: [output_dim], e.g., [128]
                3: tensor([-0.95, 0.05, ..., 0.08])   # 0号簇下所有参考集other位点(类型3)的平均嵌入向量，形状: [output_dim], e.g., [128]
            },
            'cluster_id_1': {
                0: tensor([...]),
                1: tensor([...]),
                2: tensor([...]),
                3: tensor([...])
            },
            # ... 其他cluster_id号
        }
    * site_stats   结构为 {'cluster_id': {位点类型: {'sigma2': float, 'count': int}}}
    """
    site_centers = defaultdict(dict)
    site_stats = defaultdict(dict)
    reference_residue_features = reference_residue_features.to(device)
    for ec, type_pools_indexed in tqdm(ec_site_pools_indexed_reference.items(), desc="Computing site centers", disable=True):
        for site_type, indices in type_pools_indexed.items():
            if indices.numel() == 0: continue
            embs = reference_residue_features[indices]  # embs: [N, D], 已在设备上
            center = embs.mean(dim=0)  # [D]
            site_centers[ec][site_type] = center

            if prob_mode == "gaussian":
                var = ((embs - center) ** 2).mean().item()
                site_stats[ec][site_type] = {
                    "sigma2": max(float(var), eps),
                    "count": int(embs.size(0)),
                }

    return cluster_site_type_prediction(
        site_centers=site_centers,
        site_stats=site_stats,
        query_residue_features=query_residue_features,
        id_cluster_query=id_ec_query,
        labels_dict_query=labels_dict_query,
        uid_to_offset_query=uid_to_offset_query,
        device=device,
        prob_mode=prob_mode,
        use_priors=use_priors,
        temperature=temperature,
        eps=eps
    )



@torch.no_grad()
def evaluate_site_type_prediction2(
    site_centers: Dict[str, Dict[int, torch.Tensor]],
    query_residue_features: torch.Tensor,
    id_cluster_query: Dict[str, List[str]],
    labels_dict_query: Dict[str, torch.Tensor],
    uid_to_offset_query: Dict[str, int],
    device: torch.device,
    prob_mode: str = "distance",
    temperature: float = 1.0,
) -> Tuple[Dict[str, List[float]], Dict[str, torch.Tensor]]:
    """
    Design for ClusterSiteBetaTrainer
    直接接收预计算好的 site_centers，不需要 reference 特征矩阵和索引映射。注意：此模式下默认不提供 site_stats，因此仅支持 prob_mode='distance'，或 gaussian 模式下无法使用方差。
    """
    site_stats = defaultdict(lambda: defaultdict(dict))     # 构造空的 site_stats，因为 ClusterSiteBetaTrainer 的在线累积只计算了均值
    
    return cluster_site_type_prediction(
        site_centers=site_centers,
        site_stats=site_stats,
        query_residue_features=query_residue_features,
        id_cluster_query=id_cluster_query,
        labels_dict_query=labels_dict_query,
        uid_to_offset_query=uid_to_offset_query,
        device=device,
        prob_mode=prob_mode,
        use_priors=False,
        temperature=temperature
    )



@torch.no_grad()
def cluster_site_type_prediction_binary(
    site_centers: Dict[str, Dict[int, torch.Tensor]],
    site_stats: Dict[str, Dict[int, Dict[str, float]]],
    query_residue_features: torch.Tensor,
    id_cluster_query: Dict[str, List[str]],
    labels_dict_query: Dict[str, torch.Tensor],
    uid_to_offset_query: Dict[str, int],
    device: torch.device,
    prob_mode: str = "distance",
    use_priors: bool = True,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[Dict[str, List[float]], Dict[str, torch.Tensor]]:
    """
    Binary version of cluster_site_type_prediction.
    """
    # Binary classification: num_classes = 2 (0: Non-active, 1: Active)
    num_classes = 2
    all_metrics = defaultdict(list)
    per_sample_probs: Dict[str, torch.Tensor] = {}
    query_uids = sorted(id_cluster_query.keys())
    
    for uniprot_id in tqdm(query_uids, desc="Evaluating Query Set (Binary)", disable=True):
        cluster_list = id_cluster_query[uniprot_id]
        if not cluster_list: continue
        cluster_id = cluster_list[0]

        if cluster_id not in site_centers: continue
        if uniprot_id not in labels_dict_query or uniprot_id not in uid_to_offset_query: continue

        offset = uid_to_offset_query[uniprot_id]
        true_labels = labels_dict_query[uniprot_id] # Binary labels 0/1
        seq_len = true_labels.shape[0]
        residue_embs = query_residue_features[offset : offset + seq_len].to(device)

        centers = site_centers[cluster_id]
        if not centers: continue
        
        # Center types should be 0 and/or 1
        center_types = sorted(centers.keys())
        center_vectors = torch.stack([centers[t] for t in center_types]).to(device)
        
        dist_matrix = torch.cdist(residue_embs, center_vectors, p=2) # [L, N_types]
        sq_dists = dist_matrix ** 2

        # =========================================== Top@compute per_sample_probs ======================================================
        if prob_mode == "gaussian":
            sigma2_list = []
            count_list = []
            for t in center_types:
                st = site_stats[cluster_id][t]
                sigma2_list.append(st["sigma2"])
                count_list.append(st["count"])
            sigma2 = torch.tensor(sigma2_list, device=device)
            counts = torch.tensor(count_list, dtype=torch.float32, device=device)
            if use_priors:
                priors = counts / counts.sum()
                log_priors = torch.log(priors.clamp_min(1e-12))
            else:
                log_priors = torch.zeros_like(sigma2)
            logits_sub = -0.5 * sq_dists / sigma2.clamp_min(eps) + log_priors
        
        elif prob_mode == "distance":
            logits_sub = -sq_dists

        if temperature is not None and temperature > 0: 
            logits_sub = logits_sub / float(temperature)
        
        probs_sub = torch.softmax(logits_sub, dim=1)
        
        # Map back to full [L, 2] probability matrix
        probs_full = torch.zeros(seq_len, num_classes, device=device, dtype=probs_sub.dtype)
        for i, t in enumerate(center_types):
            if t < num_classes:
                probs_full[:, t] = probs_sub[:, i]

        per_sample_probs[uniprot_id] = probs_full.detach().to(torch.float32).cpu()
        # =========================================== Bottom@compute per_sample_probs ===================================================

        # =========================================== Top@compute all_metrics ======================================================
        if prob_mode == "gaussian":
            preds = torch.argmax(probs_full, dim=1)
        elif prob_mode == "distance":
            # Distance mode: strictly follow original logic based on minimum geometric distance (argmin)
            min_dist_indices = torch.argmin(dist_matrix, dim=1)
            # Ensure preds are on the correct device
            preds = torch.tensor([center_types[i] for i in min_dist_indices.cpu()], dtype=torch.long).to(device)

        # Use valid_mask as all ones
        valid_mask = torch.ones_like(true_labels, dtype=torch.bool).unsqueeze(0).to(device)
        
        # Use compute_binary_classification_metrics for binary tasks
        sample_metrics = compute_binary_classification_metrics(
            predictions=preds.unsqueeze(0),
            probabilities=probs_full[:, 1].unsqueeze(0), # Prob of positive class
            labels=true_labels.to(device).unsqueeze(0),
            valid_mask=valid_mask,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc', 'ap', 'mcc']
        )
        
        for metric_name, values in sample_metrics.items():
            if values:
                all_metrics[metric_name].extend(values)
        # =========================================== Bottom@compute all_metrics ===================================================

    if not any(v for v in all_metrics.values()):
        print("Warning: No samples were evaluated. Check Cluster/EC overlap.")
        return {}, {} 

    return all_metrics, per_sample_probs



@torch.no_grad()
def evaluate_site_type_prediction_binary(
    reference_residue_features: torch.Tensor,
    query_residue_features: torch.Tensor,
    ec_site_pools_indexed_reference: Dict[str, Dict[int, torch.Tensor]],
    id_ec_query: Dict[str, List[str]],
    labels_dict_query: Dict[str, torch.Tensor],
    uid_to_offset_query: Dict[str, int],
    device: torch.device,
    prob_mode: str = "distance",
    use_priors: bool = True,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[Dict[str, List[float]], Dict[str, torch.Tensor]]:
    
    site_centers = defaultdict(dict)
    site_stats = defaultdict(dict)
    reference_residue_features = reference_residue_features.to(device)
    
    # Compute centers for binary types (0 and 1)
    for ec, type_pools_indexed in tqdm(ec_site_pools_indexed_reference.items(), desc="Computing binary site centers", disable=True):
        for site_type, indices in type_pools_indexed.items():
            if indices.numel() == 0: continue
            embs = reference_residue_features[indices]
            center = embs.mean(dim=0)
            site_centers[ec][site_type] = center

            if prob_mode == "gaussian":
                var = ((embs - center) ** 2).mean().item()
                site_stats[ec][site_type] = {
                    "sigma2": max(float(var), eps),
                    "count": int(embs.size(0)),
                }

    return cluster_site_type_prediction_binary(
        site_centers=site_centers,
        site_stats=site_stats,
        query_residue_features=query_residue_features,
        id_cluster_query=id_ec_query,
        labels_dict_query=labels_dict_query,
        uid_to_offset_query=uid_to_offset_query,
        device=device,
        prob_mode=prob_mode,
        use_priors=use_priors,
        temperature=temperature,
        eps=eps
    )