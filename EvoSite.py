import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
import os, time, re, pickle
import logging
import esm
from torchdrug import layers, data
from omegaconf import OmegaConf, DictConfig
import numpy as np
import random
import glob
import math
from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer


from model.esm_model import EvolutionaryScaleModeling, DCAScalingScheduler, EvolutionaryScaleModelingCombined
from model.esm_model_2 import EvolutionaryScaleModelingBeta
from model.msa_model import EvoMSATransformer
from model.structure_model import StructureEncoder
from model.rxn_model import ReactionMGMTurnNet
from common.utils import (SharedEncoder, 
                          residue_level_soft_label_alignment, 
                          DCAGraphNetwork, 
                          ConditionNet, ConditionNet2, ConditionNet3,
                          PromptAttentionPredictor, Predictior, PairBiasedAttentionPredictor,
                          GlobalMultiHeadAttention,
                          pack_residue_feats,
                          CoevolutionPredictionHead, DistancePredictionHead,
                          PhiGnetCouplingProcessor, PhiGnetGraphConv,
                          GraphECEncoder, GraphECFeatureCalculator,
                          read_model_state, load_pretrain_model_state,
                          CouplingFeatureEncoder,
                          trunc_normal_,
                          check_rxnfp
                        )
from model.model_structure.enzyme_attn_model import (
    EnzymeFusionNetworkWrapper as EnzymeAttnNetwork,
    EnzymeSaProtFusionNetworkWrapper as EnzymeAttnSaProtNetwork,
    EnzymeESMWrapper as EnzymeESMNetwork,
)

from model.model_structure.rxn_attn_model import (
    ReactionMGMTurnNet as RXNAttnNetwork,
    GlobalMultiHeadAttention,
    GELU,
)


from model.rfdiffusion.util import writepdb_multi, writepdb
from model.rfdiffusion.inference import utils as iu, symmetry
from model.rfdiffusion.Track_module import IterativeSimulator, IterBlock, Str2Str
from model.rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
from model.rfdiffusion.diffusion import Diffuser
from model.rfdiffusion import util
from model.rfdiffusion.kinematics import get_init_xyz, xyz_to_t2d
from model.rfdiffusion.chemical import seq2chars
from model.rfdiffusion.util_module import ComputeAllAtomCoords
from model.rfdiffusion.potentials.manager import PotentialManager
from model.rfdiffusion.model_input_logger import pickle_function_call

SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles


from model.network.RoseTTAFoldModel import RoseTTAFold
from model.network.kinematics import rosetta_xyz_to_t2d

from model.model_structure.enzyme_attn_model import EnzymeFusionNetworkWrapper as EnzymeAttnNetwork



class EvoSite(nn.Module):
    """
    Main EvoSite model that integrates all three stages:
    1. ESM model with DCA feature injection/alignment, MSA model
    2. Multimodal alignment of ESM and MSA features
    3. Cross-attention and DCA Graph Network/Condition Net for active site prediction


    关于special tokens的处理：
        * 第一阶段、第二阶段、第三阶段的交叉注意力层都是带有special token的(seq_len)，只不过使用valid_mask来忽略special token的影响
        * 不带special token的(valid_len)模块有：ESM2中的auxiliary_losses模式、第三阶段的DCAGraphNetwork
    
    关于class EvoSite的训练与推理：https://chatgpt.com/g/g-p-6868b3b0127881919d1be7623d1bb5b0-method/c/686e9e21-7a04-8005-a4cb-3bc6e3c59ef9
    """
    
    def __init__(
        self,
        # =========================================== Top@Stage 1 ======================================================
        # ESM model configuration
        esm_model_name: str = "esm2_t33_650M_UR50D",
        esm_pretrained_weights_path: Optional[str] = None,
        freeze_esm_backbone: bool = True,
        unfreeze_esm_layers: Optional[List[int]] = [16, 31, 32],
        
        # DCA injection/alignment options
        # 1 确保use_auxiliary_losses和use_relative_bias_injection不能同时为true
        # 2 alignment_layers \in unfreeze_esm_layers
        use_auxiliary_losses: bool = True,
        use_relative_bias_injection: bool = False,
        use_single_site_potentials: bool = False,
        alignment_loss_mode: str = "kl",
        use_concat_map: bool = True,
        attention_fusion_method: str = "weighted_average",
        alignment_layers: Optional[List[int]] = [16, 31, 32],
        
        # MSA featurizer configuration
        use_msa_featurizer: bool = True,
        msa_model_name: str = "esm_msa1b_t12_100M_UR50S",
        # =========================================== Bottom@Stage 1 =================================================== 


        # =========================================== Top@Stage 2 ====================================================== 
        # Multimodal fusion options
        fusion_method: str = "soft_label",  # "soft_label", "topology", "concat" 
        shared_encoder_layers: int = 4,
        shared_encoder_heads: int = 8,
        shared_encoder_dim: int = 768,
        use_depth_gate: bool = False,
        # =========================================== Bottom@Stage 2 =================================================== 


        # =========================================== Top@Stage 3 ======================================================
        # Cross-attention 
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.1,

        # Graph Network
        use_dca_graph: bool = False,
        dca_graph_hidden_dim: int = 256,
        dca_graph_layers: int = 3,
        dca_graph_dropout: float = 0.1,
        dca_threshold: float = 0.05,
        dca_top_k: Optional[int] = None,
        gnn_type: str = "gcn",  # 这个参数需要重点考虑

        # Condition Network
        use_condition_net: bool = True,
        condition_hidden_dim: int = 256,  
        condition_film_bottleneck: int = 64,
        condition_use_residual: bool = True,
        condition_seq_len: int = 1026,  
        condition_dca_norm_method: str = "tanh",
        
        # MLP prediction network
        classification_output_dim: int = 4,  # Binary classification for active sites by default
        dropout: float = 0.1,
        

        # Prompt-based prediction network
        use_prompt_predictor: bool = False,  # Whether to use prompt-based prediction
        prompt_predictor_mode: str = "replace",  # "replace", "ensemble", "auxiliary"
        prompt_init_method: str = "random",  # "random", "physico_chemical", "normal"
        prompt_temperature: float = 1.0,
        prompt_learnable_temperature: bool = True,
        prompt_use_multi_head_attention: bool = False,
        prompt_attention_heads: int = 8,
        prompt_attention_dropout: float = 0.1,
        prompt_use_layer_norm: bool = True,
        prompt_regularization: float = 0.01,
        prompt_use_diversity_loss: bool = True,
        prompt_diversity_loss_weight: float = 0.1,
        prompt_similarity_metric: str = "dot_product",  # "dot_product", "cosine", "scaled_dot_product"
        prompt_use_residual: bool = False,
        prompt_dropout: float = 0.1,
        ensemble_weight: float = 0.5,  # Weight for ensemble between MLP and prompt predictor
        # =========================================== Bottom@Stage 3 ===================================================

        
        # General options
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        
        # DCA耦合矩阵偏置缩放因子
        use_dynamic_dca_scaling: bool = True,
        dynamic_scaling_strategy: str = "cosine_decay",
        initial_scaling_factor: float = 2.0,
        final_scaling_factor: float = 0.5,
    ):
        """
        Initialize the EvoSite model with all its components.
        调参：https://chatgpt.com/c/686d588c-adc0-8005-84ac-bd6484927d7e
        """
        super(EvoSite, self).__init__()
        self._device = device
        
        # Save configuration
        self.use_auxiliary_losses = use_auxiliary_losses
        self.use_concat_map = use_concat_map
        self.use_relative_bias_injection = use_relative_bias_injection
        self.use_single_site_potentials = use_single_site_potentials
        self.use_msa_featurizer = use_msa_featurizer
        self.fusion_method = fusion_method
        self.use_depth_gate = use_depth_gate
        self.use_dca_graph = use_dca_graph
        self.use_condition_net = use_condition_net
        self.classification_output_dim = classification_output_dim
        self.use_dynamic_dca_scaling = use_dynamic_dca_scaling
        self.initial_scaling_factor = initial_scaling_factor
        self.use_prompt_predictor = use_prompt_predictor
        self.prompt_predictor_mode = prompt_predictor_mode
        self.ensemble_weight = ensemble_weight
        
        
        # Initialize ESM model with DCA alignment capabilities
        self.esm_model = EvolutionaryScaleModeling(
            model_name=esm_model_name,
            pretrained_weights_path=esm_pretrained_weights_path,
            freeze_backbone=freeze_esm_backbone,
            unfreeze_layers=unfreeze_esm_layers,
            use_auxiliary_losses=use_auxiliary_losses,
            use_concat_map = use_concat_map,
            use_relative_bias_injection=use_relative_bias_injection,
            use_single_site_potentials=use_single_site_potentials,
            alignment_loss_mode=alignment_loss_mode,
            alignment_layers=alignment_layers,
            attention_fusion_method=attention_fusion_method,
            device=device
        )
        
        # Get ESM embedding dimension
        self.esm_hidden_dim = self.esm_model.hidden_dim
        
        # Initialize MSA featurizer if enabled
        if use_msa_featurizer:
            self.msa_model = EvoMSATransformer(
                model_path=msa_model_name,
                use_cuda=(device != "cpu"),
                device=device
            )
            
            self.msa_hidden_dim = self.msa_model.msa_out_dim
            
            self.shared_encoder = SharedEncoder(
                esm_dim=self.esm_hidden_dim,
                msa_dim=self.msa_hidden_dim,
                hidden_dim=shared_encoder_dim,
                num_layers=shared_encoder_layers,
                num_heads=shared_encoder_heads,
                dropout=dropout
            )

            self.fusion_dim = shared_encoder_dim
            
            # Optional depth gate for adaptive fusion
            if use_depth_gate:
                self.depth_gate = nn.Sequential(
                    nn.Linear(1, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
                
            # Cross-attention layer for MSA-to-ESM feature fusion
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.fusion_dim,
                num_heads=cross_attention_heads,
                dropout=cross_attention_dropout,
                batch_first=True
            )

        else:
            # No MSA features, use ESM embeddings directly
            self.fusion_dim = self.esm_hidden_dim
        
        # Initialize DCA Graph Network if enabled
        if use_dca_graph:
            self.dca_graph_network = DCAGraphNetwork(
                input_dim=self.fusion_dim,
                hidden_dim=dca_graph_hidden_dim,
                output_dim=self.fusion_dim,
                num_layers=dca_graph_layers,
                dropout=dca_graph_dropout,
                threshold=dca_threshold,
                top_k=dca_top_k,
                gnn_type=gnn_type,
                use_batch_norm=True,
                add_residual=True
            )
            
            # Projection layer for concatenated features
            self.graph_projection = nn.Linear(self.fusion_dim * 2, self.fusion_dim)

        # Initialize Condition Network if enabled
        elif use_condition_net:
            self.condition_network = ConditionNet(
                fusion_dim=self.fusion_dim,
                output_dim=None,  # 默认使用fusion_dim
                dropout=dropout,
                film_bottleneck=condition_film_bottleneck,
                synergy_hidden_dim=condition_hidden_dim,  
                global_max_seq_len=condition_seq_len,  
                use_residual=condition_use_residual,
                dca_norm_method=condition_dca_norm_method
            )
    
        # Initialize prediction heads
        if use_prompt_predictor:
            # Initialize prompt-based predictor
            self.prompt_predictor = PromptAttentionPredictor(
                input_dim=self.fusion_dim,
                num_site_types=classification_output_dim,
                prompt_init_method=prompt_init_method,
                temperature=prompt_temperature,
                learnable_temperature=prompt_learnable_temperature,
                use_multi_head_attention=prompt_use_multi_head_attention,
                num_attention_heads=prompt_attention_heads,
                attention_dropout=prompt_attention_dropout,
                use_layer_norm=prompt_use_layer_norm,
                prompt_regularization=prompt_regularization,
                use_prompt_diversity_loss=prompt_use_diversity_loss,
                diversity_loss_weight=prompt_diversity_loss_weight,
                similarity_metric=prompt_similarity_metric,
                use_residual_connection=prompt_use_residual,
                dropout=prompt_dropout
            )
            
            # Initialize traditional MLP predictor if in ensemble mode
            if prompt_predictor_mode == "ensemble":
                self.mlp_predictor = nn.Sequential(
                    nn.Linear(self.fusion_dim, self.fusion_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.fusion_dim * 2, classification_output_dim)
                )
        else:
            # Initialize MLP prediction head only
            self.prediction_head = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.fusion_dim * 2, classification_output_dim)
            )
        
        # Initialize DCA scaling scheduler if enabled
        if self.use_dynamic_dca_scaling:
            self.dca_scaling_scheduler = DCAScalingScheduler(
                initial_scale=initial_scaling_factor,
                final_scale=final_scaling_factor,
                total_steps=1000,  # Default value, will be updated dynamically during training
                strategy=dynamic_scaling_strategy
            )
        else:
            self.dca_scaling_scheduler = None


        # Move model to device
        self.to(device)
    


    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, List[torch.Tensor], List[str], List[int]]],
        esm_repr_layers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the complete EvoSite model.
        
        Args:
            batch: Dictionary output from collate_enzyme_msa_batch containing
            esm_repr_layers: Optional[List[int]]，List of ESM transformer layers to extract representations from. 注意repr_layers的索引系统是1-based for transformer layers，即第一个transformer层索引是1，最后一个transformer层的索引是num_layers

        Returns:
            Dictionary containing results including predictions, losses, and optionally embeddings

                'esm_embeddings': torch.Tensor，第一阶段结束后，ESM2模型输出的残基级嵌入 [batch_size, seq_len, hidden_dim]，包含所有token（含special token）
                'esm_valid_tokens_mask': torch.BoolTensor，有效残基掩码 [batch_size, seq_len]，1表示有效残基（非special token），0为special token
                'esm_valid_residue_indices': List[torch.LongTensor]，每个样本有效残基的位置索引，长度batch_size，每个元素形状[valid_len_i]
                'msa_embeddings': torch.Tensor (仅当use_msa_featurizer=True且输入有msa_tokens)，MSA Transformer输出的残基级嵌入 [batch_size, seq_len, msa_hidden_dim]

                'esm_encoded': torch.Tensor (仅当use_msa_featurizer=True)，共享编码器输出的ESM特征 [batch_size, seq_len, fusion_dim]
                'msa_encoded': torch.Tensor (仅当use_msa_featurizer=True)，共享编码器输出的MSA特征 [batch_size, seq_len, fusion_dim]
                'residue_level_soft_label_loss': torch.Tensor (仅当fusion_method="soft_label"且use_msa_featurizer=True)，残基级软标签对齐损失，float标量

                'cross_attention_output_embeddings': torch.Tensor (仅当use_msa_featurizer=True)，交叉注意力融合后的残基特征 [batch_size, seq_len, fusion_dim]
                'refined_embeddings': torch.Tensor (仅当 use_dca_graph=True 或 use_condition_net=True)，经图网络或条件网络处理后的特征 [batch_size, seq_len, fusion_dim]

                'logits': torch.Tensor，最终分类头输出的logits [batch_size, seq_len, classification_output_dim]，用于活性位点类型或位置预测

                # Prompt-based prediction outputs (when use_prompt_predictor=True)
                'prompt_logits': torch.Tensor (仅当use_prompt_predictor=True)，提示词预测器输出的logits [batch_size, seq_len, classification_output_dim]
                'prompt_similarities': torch.Tensor (仅当use_prompt_predictor=True且不使用multi-head attention)，残基与提示词的相似度 [batch_size, seq_len, classification_output_dim]
                'prompt_regularization_loss': torch.Tensor (仅当use_prompt_predictor=True)，提示词正则化损失，float标量
                'prompt_diversity_loss': torch.Tensor (仅当use_prompt_predictor=True且use_prompt_diversity_loss=True)，提示词多样性损失，float标量
                'mlp_logits': torch.Tensor (仅当prompt_predictor_mode="ensemble")，MLP预测器输出的logits [batch_size, seq_len, classification_output_dim]

                'auxiliary_loss': torch.Tensor (仅当use_auxiliary_losses=True且esm_outputs含coupling_losses)，ESM与DCA耦合矩阵的辅助对齐损失，float标量
                'attention_maps': List[torch.Tensor] (仅当use_auxiliary_losses=True且use_concat_map=False且esm_outputs含attention_maps)，ESM注意力图（去除special token），长度batch_size，每个元素[valid_len_i, valid_len_i]
                'contact_maps': List[torch.Tensor] (仅当use_auxiliary_losses=True且use_concat_map=True且esm_outputs含contact_maps)，ESM接触图（去除special token），长度batch_size，每个元素[valid_len_i, valid_len_i]

                'single_site_loss': torch.Tensor (仅当use_single_site_potentials=True且esm_outputs含single_site_loss)，单点势KL损失，float标量
                'range_penalty_loss': torch.Tensor (仅当self.training==True且esm_outputs含range_penalty_loss)，范围正则损失，float标量
                'esm_repr_embeddings': Dict[int, torch.Tensor] (仅当esm_repr_layers不为空)，ESM指定层的隐藏特征，键为层号，值为[batch_size, seq_len, hidden_dim]
        """
        results = {}

        tokens = batch['sequences']
        msa_tokens = batch['msa']
        dca_coupling_matrix = batch['coupling_matrices']
        single_site_potentials = batch['single_site_potentials']
        conservation_scores = batch.get('conservation_scores', None)        # 预期输入[batch_size, seq_len]


        if self.use_dynamic_dca_scaling and self.dca_scaling_scheduler is not None:
            if self.training:
                current_scaling = self.dca_scaling_scheduler.get_scale()
            else:
                current_scaling = self.dca_scaling_scheduler.final_scale.item()
        else:
            current_scaling = 1.0  
        
        esm_outputs = self.esm_model(
            tokens=tokens,
            sequences=None,  
            dca_coupling_matrix=dca_coupling_matrix,
            single_site_potentials=single_site_potentials,
            repr_layers=esm_repr_layers,
            dynamic_scaling_factor=current_scaling
        )
        
        # =========================================== Top@fetch from esm_outputs ====================================================== 
        esm_embeddings = esm_outputs['embeddings']  # [batch_size, seq_len, hidden_dim]
        esm_valid_tokens_mask = esm_outputs.get('valid_tokens_mask', None)
        esm_valid_residue_indices = esm_outputs.get('valid_residue_indices', None)
        
        results['esm_embeddings'] = esm_outputs['embeddings']   # [batch_size, seq_len, hidden_dim]
        results['esm_valid_tokens_mask'] = esm_valid_tokens_mask  # [batch_size, seq_len]
        results['esm_valid_residue_indices'] = esm_valid_residue_indices  

        if self.use_auxiliary_losses and 'coupling_losses' in esm_outputs:
            results['auxiliary_loss'] = esm_outputs['coupling_losses']['coupling_loss']
        if self.use_auxiliary_losses  and not self.use_concat_map and 'attention_maps' in esm_outputs:
            results['attention_maps'] = esm_outputs['attention_maps']
        if self.use_auxiliary_losses and self.use_concat_map and 'contact_maps' in esm_outputs:
            results['contact_maps'] = esm_outputs['contact_maps']
        if esm_repr_layers:
            results['esm_repr_embeddings'] = esm_outputs['representations']
        if self.use_single_site_potentials and 'single_site_loss' in esm_outputs:
            results['single_site_loss'] = esm_outputs['single_site_loss']
        if self.training and 'range_penalty_loss' in esm_outputs:
            results['range_penalty_loss'] = esm_outputs['range_penalty_loss']
        # =========================================== Bottom@fetch from esm_outputs =================================================== 
        
        if self.use_msa_featurizer and msa_tokens is not None:
            msa_embeddings, msa_valid_tokens_mask, msa_valid_residue_indices = self.msa_model(msa_tokens)           
            results['msa_embeddings'] = msa_embeddings
            

            # 通过共享编码器处理
            # esm_valid_tokens_mask 与 msa_valid_tokens_mask的规则是1表示有效残基，0表示special token
            # 但是self.shared_encoder的forward函数预期的输入forward(self, esm_features, msa_features, esm_valid_tokens_mask, msa_valid_tokens_mask)中esm_valid_tokens_mask, msa_valid_tokens_mask的规则是1表示需要被屏蔽的位置，0表示有效残基，因此需要取反
            esm_encoded, msa_encoded = self.shared_encoder(esm_embeddings, msa_embeddings, ~esm_valid_tokens_mask, ~msa_valid_tokens_mask)
            results['esm_encoded'] = esm_encoded
            results['msa_encoded'] = msa_encoded
            
            # 进一步对齐
            if self.fusion_method == "soft_label":
                # 计算软标签对齐损失
                residue_level_soft_label_loss = residue_level_soft_label_alignment(
                    esm_embeddings=esm_encoded,
                    msa_embeddings=msa_encoded,
                    esm_mask=esm_valid_tokens_mask,  
                    msa_mask=msa_valid_tokens_mask,  
                    temperature=0.5,
                    loss_weight=1.0
                )
                results['residue_level_soft_label_loss'] = residue_level_soft_label_loss

            elif self.fusion_method == "topology":
                print("need to implement topology fusion method")

            
            # 交叉注意力层，使用ESM特征作为查询，MSA特征作为键和值
            cross_attn_output, cross_attn_weights = self.cross_attention(
                query=esm_encoded,
                key=msa_encoded,
                value=msa_encoded,
                key_padding_mask=~msa_valid_tokens_mask  # key_padding_mask参数的规则是：True表示要忽略的位置
            )
            
            fused_embeddings = cross_attn_output
            results['cross_attention_output_embeddings'] = fused_embeddings


        else:
            # 没有MSA特征，直接使用ESM嵌入
            fused_embeddings = esm_embeddings
        

        
        # 使用DCA图网络处理
        if self.use_dca_graph and dca_coupling_matrix is not None:
            # 确保dca_coupling_matrix是列表
            coupling_matrices = dca_coupling_matrix if isinstance(dca_coupling_matrix, list) else [dca_coupling_matrix]
            graph_embeddings = self.dca_graph_network(
                residue_embeddings=fused_embeddings,
                dca_matrices=coupling_matrices,
                valid_residue_indices=esm_valid_residue_indices
            )
            # 与原始嵌入拼接并投影
            combined_embeddings = torch.cat([fused_embeddings, graph_embeddings], dim=-1)
            refined_embeddings = self.graph_projection(combined_embeddings)
            results['refined_embeddings'] = refined_embeddings
        
        # 使用条件网络处理
        elif self.use_condition_net and single_site_potentials is not None:
            # if conservation_scores is None:
            #     batch_size, seq_len, _ = fused_embeddings.shape
            #     conservation_scores = torch.zeros((batch_size, seq_len), device=fused_embeddings.device)
            
            # # 处理single_site_potentials
            # if isinstance(single_site_potentials, list):
            #     # 需要处理不同长度的single_site_potentials
            #     # 这里假设每个batch项的valid_residue_indices与single_site_potentials对应
            #     processed_potentials = []
            #     for i, (potentials, indices) in enumerate(zip(single_site_potentials, esm_valid_residue_indices)):
            #         # 创建全零张量，大小与fused_embeddings[i]相同
            #         full_potentials = torch.zeros(
            #             (fused_embeddings.shape[1], potentials.shape[1]), 
            #             device=fused_embeddings.device
            #         )
            #         # 将potentials填入valid位置
            #         full_potentials[indices] = potentials
            #         processed_potentials.append(full_potentials)
                
            #     # 堆叠处理后的tensors
            #     batch_potentials = torch.stack(processed_potentials)
            # else:
            #     # 如果已经是批次张量，直接使用
            #     batch_potentials = single_site_potentials
            
            # 使用条件网络处理融合特征
            refined_embeddings = self.condition_network(
                fused_embeddings=fused_embeddings,
                conservation_scores=conservation_scores,
                dca_single_potentials=single_site_potentials,
                coupling_matrix=dca_coupling_matrix, 
                valid_residue_indices=esm_valid_residue_indices, 
                valid_tokens_mask=esm_valid_tokens_mask  
            )
            results['refined_embeddings'] = refined_embeddings
            
        else:
            # 没有图处理或条件网络，直接使用融合嵌入
            refined_embeddings = fused_embeddings
        
        # =========================================== Top@Prediction ====================================================== 
        if self.use_prompt_predictor:
            if self.prompt_predictor_mode == "replace":
                # Use only prompt-based predictor
                prompt_results = self.prompt_predictor(
                    residue_embeddings=refined_embeddings,
                    valid_mask=esm_valid_tokens_mask,
                    return_attention_weights=True
                )
                results['logits'] = prompt_results['logits']
                results['prompt_logits'] = prompt_results['logits']
                
                # Add prompt-specific outputs
                if 'prompt_similarities' in prompt_results:
                    results['prompt_similarities'] = prompt_results['prompt_similarities']
                if 'attention_weights' in prompt_results:
                    results['prompt_attention_weights'] = prompt_results['attention_weights']
                results['prompt_regularization_loss'] = prompt_results['prompt_regularization_loss']
                if 'prompt_diversity_loss' in prompt_results:
                    results['prompt_diversity_loss'] = prompt_results['prompt_diversity_loss']
                    
            elif self.prompt_predictor_mode == "ensemble":
                # Use ensemble of prompt predictor and MLP
                prompt_results = self.prompt_predictor(
                    residue_embeddings=refined_embeddings,
                    valid_mask=esm_valid_tokens_mask,
                    return_attention_weights=True
                )
                mlp_logits = self.mlp_predictor(refined_embeddings)
                
                # Ensemble the predictions
                ensemble_logits = (
                    self.ensemble_weight * prompt_results['logits'] + 
                    (1 - self.ensemble_weight) * mlp_logits
                )
                
                results['logits'] = ensemble_logits
                results['prompt_logits'] = prompt_results['logits']
                results['mlp_logits'] = mlp_logits
                
                # Add prompt-specific outputs
                if 'prompt_similarities' in prompt_results:
                    results['prompt_similarities'] = prompt_results['prompt_similarities']
                if 'attention_weights' in prompt_results:
                    results['prompt_attention_weights'] = prompt_results['attention_weights']
                results['prompt_regularization_loss'] = prompt_results['prompt_regularization_loss']
                if 'prompt_diversity_loss' in prompt_results:
                    results['prompt_diversity_loss'] = prompt_results['prompt_diversity_loss']
                    
            elif self.prompt_predictor_mode == "auxiliary":
                # Use MLP as main predictor, prompt as auxiliary
                main_logits = self.prediction_head(refined_embeddings)
                prompt_results = self.prompt_predictor(
                    residue_embeddings=refined_embeddings,
                    valid_mask=esm_valid_tokens_mask,
                    return_attention_weights=True
                )
                
                results['logits'] = main_logits
                results['prompt_logits'] = prompt_results['logits']
                
                # Add prompt-specific outputs for auxiliary loss computation
                if 'prompt_similarities' in prompt_results:
                    results['prompt_similarities'] = prompt_results['prompt_similarities']
                if 'attention_weights' in prompt_results:
                    results['prompt_attention_weights'] = prompt_results['attention_weights']
                results['prompt_regularization_loss'] = prompt_results['prompt_regularization_loss']
                if 'prompt_diversity_loss' in prompt_results:
                    results['prompt_diversity_loss'] = prompt_results['prompt_diversity_loss']
        else:
            # Use traditional MLP prediction head only
            logits = self.prediction_head(refined_embeddings)
            results['logits'] = logits
        # =========================================== Bottom@Prediction =================================================== 
        
        return results



class EnzymeActiveSiteClsModel(nn.Module):
    def __init__(
        self,
        rxn_model_path,
        num_active_site_type,
        from_scratch=False,
        use_saprot_esm=False,
    ) -> None:
        super(EnzymeActiveSiteClsModel, self).__init__()
        self.from_scratch = from_scratch
        self.num_active_site_type = num_active_site_type

        if not use_saprot_esm:
            self.enzyme_attn_model = EnzymeAttnNetwork(
                use_graph_construction_model=True
            )
        else:
            self.enzyme_attn_model = EnzymeAttnSaProtNetwork(
                use_graph_construction_model=True
            )

        self.rxn_attn_model = self._load_rxn_attn_model(rxn_model_path)

        self.brige_model = Predictior(
            self.enzyme_attn_model.output_dim, self.rxn_attn_model.node_out_dim
        )

        self.interaction_net = GlobalMultiHeadAttention(
            self.rxn_attn_model.node_out_dim,
            heads=8,
            n_layers=3,
            cross_attn_h_rate=1,
            dropout=0.1,
            positional_number=0,
        )

        self.active_site_type_net = Predictior(
            self.rxn_attn_model.node_out_dim, num_active_site_type
        )

    def forward(self, batch, return_features: bool = False):
        output_protein = self.enzyme_attn_model(batch)
        output_reaction = self.rxn_attn_model(
            **batch["reaction_graph"], return_rts=True
        )

        substrate_node_feature, substrate_mask = output_reaction

        protein_node_feature = self.brige_model(output_protein["node_feature"])

        protein_node_feature, protein_mask = pack_residue_feats(
            batch["protein_graph"], protein_node_feature
        )

        protein_node_feature, _, _, _, _ = self.interaction_net(
            src=protein_node_feature,
            tgt=substrate_node_feature,
            src_mask=protein_mask,
            tgt_mask=substrate_mask,
        )
        if return_features:
            return protein_node_feature, protein_mask
        protein_node_feature = protein_node_feature[protein_mask.bool()]

        out = self.active_site_type_net(protein_node_feature)

        return out, protein_mask

    def _load_rxn_attn_model(self, model_state_path):
        model_state, rxn_attn_args = read_model_state(model_save_path=model_state_path)

        rxn_attn_model = RXNAttnNetwork(
            node_in_feats=rxn_attn_args["node_in_feats"],
            node_out_feats=rxn_attn_args["node_out_feats"],
            edge_in_feats=rxn_attn_args["edge_in_feats"],
            edge_hidden_feats=rxn_attn_args["edge_hidden_feats"],
            num_step_message_passing=rxn_attn_args["num_step_message_passing"],
            attention_heads=rxn_attn_args["attention_heads"],
            attention_layers=rxn_attn_args["attention_layers"],
            dropout=rxn_attn_args["dropout"],
            num_atom_types=rxn_attn_args["num_atom_types"],
            cross_attn_h_rate=rxn_attn_args["cross_attn_h_rate"],
            is_pretrain=False,
        )
        if not self.from_scratch:
            print(
                f"Loading reaction attention model checkpoint from {model_state_path}..."
            )
            rxn_attn_model = load_pretrain_model_state(rxn_attn_model, model_state)
        else:
            pass
            # print("Reaction attention model from scratch...")
        return rxn_attn_model
    


class EvoSiteCombined(EvoSite):
    """
    EvoSite model with combined bias injection and auxiliary losses.
    
    This class extends the main EvoSite model to simultaneously use both bias injection
    and attention map KL loss, overcoming the mutual exclusion in the original design.
    """
    
    def __init__(
        self,
        # =========================================== Top@Stage 1 ======================================================             
        # ESM model configuration
        esm_model_name: str = "esm2_t33_650M_UR50D",
        esm_pretrained_weights_path: Optional[str] = None,
        freeze_esm_backbone: bool = True,
        unfreeze_esm_layers: Optional[List[int]] = [16, 31, 32],
        
        # DCA injection/alignment options
        use_auxiliary_losses: bool = True,      # Enable attention map KL loss
        use_relative_bias_injection: bool = True,  # Enable bias injection
        use_single_site_potentials: bool = False,
        alignment_loss_mode: str = "kl",
        use_concat_map: bool = False,  # Use attention maps 
        attention_fusion_method: str = "weighted_average",
        alignment_layers: Optional[List[int]] = [16, 31, 32],
        
        # MSA featurizer configuration
        use_msa_featurizer: bool = True,
        msa_model_name: str = "esm_msa1b_t12_100M_UR50S",
        # =========================================== Bottom@Stage 1 =================================================== 
        
        # =========================================== Top@Stage 2 ====================================================== 
        # Multimodal fusion options
        fusion_method: str = "soft_label",  # "soft_label", "topology", "concat" 
        shared_encoder_layers: int = 4,
        shared_encoder_heads: int = 8,
        shared_encoder_dim: int = 768,
        use_depth_gate: bool = False,
        # =========================================== Bottom@Stage 2 =================================================== 

        # =========================================== Top@Stage 3 ======================================================
        # Cross-attention 
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.1,

        # Graph Network
        use_dca_graph: bool = False,
        dca_graph_hidden_dim: int = 256,
        dca_graph_layers: int = 3,
        dca_graph_dropout: float = 0.1,
        dca_threshold: float = 0.05,
        dca_top_k: Optional[int] = None,
        gnn_type: str = "gcn",

        # Condition Network
        use_condition_net: bool = True,
        condition_hidden_dim: int = 256,  
        condition_film_bottleneck: int = 64,
        condition_use_residual: bool = True,
        condition_dca_norm_method: str = "tanh",
        
        # MLP prediction network
        classification_output_dim: int = 4,
        dropout: float = 0.1,

        # Prompt-based prediction network
        use_prompt_predictor: bool = False,
        prompt_predictor_mode: str = "replace",
        prompt_init_method: str = "random",
        prompt_temperature: float = 1.0,
        prompt_learnable_temperature: bool = True,
        prompt_use_multi_head_attention: bool = False,
        prompt_attention_heads: int = 8,
        prompt_attention_dropout: float = 0.1,
        prompt_use_layer_norm: bool = True,
        prompt_regularization: float = 0.01,
        prompt_use_diversity_loss: bool = True,
        prompt_diversity_loss_weight: float = 0.1,
        prompt_similarity_metric: str = "dot_product",
        prompt_use_residual: bool = False,
        prompt_dropout: float = 0.1,
        ensemble_weight: float = 0.5,
        # =========================================== Bottom@Stage 3 ===================================================

        # General options
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        
        # DCA耦合矩阵偏置缩放因子
        use_dynamic_dca_scaling: bool = True,
        dynamic_scaling_strategy: str = "cosine_decay",
        initial_scaling_factor: float = 2.0,
        final_scaling_factor: float = 0.5,
    ):
        """
        Initialize the combined EvoSite model.
        
        Args:
            use_auxiliary_losses: Enable auxiliary losses (attention map KL loss)
            use_relative_bias_injection: Enable bias injection
            use_concat_map: Use contact maps vs attention maps for auxiliary losses
            All other args: Same as original EvoSite model
        """
        nn.Module.__init__(self)


        self._device = device
        
        # Save configuration
        self.use_auxiliary_losses = use_auxiliary_losses
        self.use_concat_map = use_concat_map
        self.use_relative_bias_injection = use_relative_bias_injection
        self.use_single_site_potentials = use_single_site_potentials
        self.use_msa_featurizer = use_msa_featurizer
        self.fusion_method = fusion_method
        self.use_depth_gate = use_depth_gate
        self.use_dca_graph = use_dca_graph
        self.use_condition_net = use_condition_net
        self.classification_output_dim = classification_output_dim
        self.use_dynamic_dca_scaling = use_dynamic_dca_scaling
        self.initial_scaling_factor = initial_scaling_factor
        
        # Save prompt predictor configuration
        self.use_prompt_predictor = use_prompt_predictor
        self.prompt_predictor_mode = prompt_predictor_mode
        self.ensemble_weight = ensemble_weight
        
        # Instantiate combined ESM model
        self.esm_model = EvolutionaryScaleModelingCombined(
            model_name=esm_model_name,
            pretrained_weights_path=esm_pretrained_weights_path,
            freeze_backbone=freeze_esm_backbone,
            unfreeze_layers=unfreeze_esm_layers,
            use_auxiliary_losses=use_auxiliary_losses,
            use_relative_bias_injection=use_relative_bias_injection,
            use_concat_map=use_concat_map,
            use_single_site_potentials=use_single_site_potentials,
            alignment_loss_mode=alignment_loss_mode,
            alignment_layers=alignment_layers,
            attention_fusion_method=attention_fusion_method,
            device=device
        )
        
        self.esm_hidden_dim = self.esm_model.hidden_dim
        
        # Initialize MSA featurizer if enabled
        if use_msa_featurizer:
            self.msa_model = EvoMSATransformer(
                model_path=msa_model_name,
                use_cuda=(device != "cpu"),
                device=device
            )
            
            self.msa_hidden_dim = self.msa_model.msa_out_dim
            
            self.shared_encoder = SharedEncoder(
                esm_dim=self.esm_hidden_dim,
                msa_dim=self.msa_hidden_dim,
                hidden_dim=shared_encoder_dim,
                num_layers=shared_encoder_layers,
                num_heads=shared_encoder_heads,
                dropout=dropout
            )

            self.fusion_dim = shared_encoder_dim
            
            # Optional depth gate for adaptive fusion
            if use_depth_gate:
                self.depth_gate = nn.Sequential(
                    nn.Linear(1, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                
            # Cross-attention layer for MSA-to-ESM feature fusion
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.fusion_dim,
                num_heads=cross_attention_heads,
                dropout=cross_attention_dropout,
                batch_first=True
            )

        else:
            # No MSA features, use ESM embeddings directly
            self.fusion_dim = self.esm_hidden_dim
        
        # Initialize DCA Graph Network if enabled
        if use_dca_graph:
            self.dca_graph_network = DCAGraphNetwork(
                input_dim=self.fusion_dim,
                hidden_dim=dca_graph_hidden_dim,
                output_dim=self.fusion_dim,
                num_layers=dca_graph_layers,
                dropout=dca_graph_dropout,
                threshold=dca_threshold,
                top_k=dca_top_k,
                gnn_type=gnn_type,
                use_batch_norm=True,
                add_residual=True
            )
            
            # Projection layer for concatenated features
            self.graph_projection = nn.Linear(self.fusion_dim * 2, self.fusion_dim)

        # Initialize Condition Network if enabled
        elif use_condition_net:
            self.condition_network = ConditionNet2(
                fusion_dim=self.fusion_dim,
                output_dim=None,  # 默认使用fusion_dim
                dropout=dropout,
                synergy_hidden_dim=condition_hidden_dim,
                synergy_attention_heads=8,
                use_residual=condition_use_residual,
                use_relative_pos_in_synergy=True  # Enable relative position in synergy
            )
    
        # Initialize prediction heads
        if use_prompt_predictor:
            # Initialize prompt-based predictor
            self.prompt_predictor = PromptAttentionPredictor(
                input_dim=self.fusion_dim,
                num_site_types=classification_output_dim,
                prompt_init_method=prompt_init_method,
                temperature=prompt_temperature,
                learnable_temperature=prompt_learnable_temperature,
                use_multi_head_attention=prompt_use_multi_head_attention,
                num_attention_heads=prompt_attention_heads,
                attention_dropout=prompt_attention_dropout,
                use_layer_norm=prompt_use_layer_norm,
                prompt_regularization=prompt_regularization,
                use_prompt_diversity_loss=prompt_use_diversity_loss,
                diversity_loss_weight=prompt_diversity_loss_weight,
                similarity_metric=prompt_similarity_metric,
                use_residual_connection=prompt_use_residual,
                dropout=prompt_dropout
            )
            
            # Initialize traditional MLP predictor if in ensemble mode
            if prompt_predictor_mode == "ensemble":
                self.mlp_predictor = nn.Sequential(
                    nn.Linear(self.fusion_dim, self.fusion_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.fusion_dim * 2, classification_output_dim)
                )
        else:
            # Initialize MLP prediction head only
            self.prediction_head = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.fusion_dim * 2, classification_output_dim)
            )
        
        # Initialize DCA scaling scheduler if enabled
        if self.use_dynamic_dca_scaling:
            self.dca_scaling_scheduler = DCAScalingScheduler(
                initial_scale=initial_scaling_factor,
                final_scale=final_scaling_factor,
                total_steps=1000,  # Default value, will be updated dynamically during training
                strategy=dynamic_scaling_strategy
            )
        else:
            self.dca_scaling_scheduler = None

        # Move model to device
        self.to(device)



    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, List[torch.Tensor], List[str], List[int]]],
        esm_repr_layers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:

        results = super().forward(batch, esm_repr_layers)
        
        # Keep only final layer representation for inference to save memory
        if not self.training and 'representations' in results:
            final_layer = max(results['representations'].keys()) if results['representations'] else None
            if final_layer is not None:
                final_repr = results['representations'][final_layer]
                results['representations'] = {final_layer: final_repr}
        
        return results



class ESMResidueClassifier(nn.Module):
    """
    ESM-2 features + MLP baseline for enzyme active site prediction.
    
    This model serves as a lightweight baseline using only ESM-2 pretrained features
    with a simple MLP classifier head. The ESM backbone is frozen to focus on
    evaluating the classification capability of ESM features alone.
    
    Architecture:
    - ESM-2 backbone (frozen): Extract residue-level representations
    - MLP classifier head: Two-layer MLP with ReLU and dropout
    """
    
    def __init__(
        self,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        num_classes: int = 4,
        hidden_dim_multiplier: int = 2,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize ESM-2 baseline classifier.

        支持不同类型的ESM2
        
        Args:
            esm_model_name: Name of ESM-2 model to use
            num_classes: Number of output classes (4 for EvoSite: background, binding, catalytic, other)
            hidden_dim_multiplier: Multiplier for hidden dimension in MLP
            dropout: Dropout rate for MLP
            device: Device to run model on
        """
        super(ESMResidueClassifier, self).__init__()
        
        self._device = device
        self.num_classes = num_classes
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading ESM-2 model: {esm_model_name}")
        
        try:
            self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
            self.logger.info(f"Successfully loaded ESM model with {self.esm_model.num_layers} layers")
        except Exception as e:
            self.logger.error(f"Failed to load ESM-2 model: {e}")
            raise
        
        # Freeze ESM backbone - only train classifier head
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        self.logger.info("Frozen all ESM-2 parameters")
        
        # Get ESM embedding dimension
        self.esm_embed_dim = self.esm_model.embed_dim
        self.logger.info(f"ESM embedding dimension: {self.esm_embed_dim}")
        
        # Build MLP classifier head
        hidden_dim = self.esm_embed_dim * hidden_dim_multiplier
        self.classifier = nn.Sequential(
            nn.Linear(self.esm_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Move to device
        self.to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Frozen parameters: {total_params - trainable_params:,}")



    def forward(
        self, 
        tokens: torch.Tensor, 
        valid_mask: torch.Tensor,
        **kwargs  # Accept additional unused arguments for compatibility
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ESM-2 + MLP classifier.
        
        Args:
            tokens: Tokenized protein sequences [batch_size, seq_len]
            valid_mask: Mask for valid (non-special) tokens [batch_size, seq_len]
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Dictionary containing:
                - logits: Classification logits [batch_size, seq_len, num_classes]
                - valid_mask: Input valid mask [batch_size, seq_len]
                - esm_features: ESM representations [batch_size, seq_len, esm_embed_dim]
        """
        batch_size, seq_len = tokens.shape
        
        # Extract ESM-2 features (last layer representations)
        with torch.no_grad():  # ESM is frozen
            esm_outputs = self.esm_model(
                tokens, 
                repr_layers=[self.esm_model.num_layers],
                return_contacts=False
            )

            # Get last layer representations，Shape: [batch_size, seq_len, esm_embed_dim]
            esm_features = esm_outputs["representations"][self.esm_model.num_layers]
        
        # Apply MLP classifier
        logits = self.classifier(esm_features)
        # Shape: [batch_size, seq_len, num_classes]
        
        return {
            "logits": logits,
            "valid_mask": valid_mask,
            "esm_features": esm_features,
        }



class MSATransformerClassifier(nn.Module):
    """
    MSA Transformer features + MLP baseline for enzyme active site prediction.
    
    This model serves as a lightweight baseline using only MSA Transformer pretrained
    features with a simple MLP classifier head. The MSA Transformer backbone is frozen
    to focus on evaluating the classification capability of MSA features alone.
    
    Architecture:
    - MSA Transformer backbone (frozen): Extract MSA-aware residue representations
    - MLP classifier head: Two-layer MLP with ReLU and dropout
    """
    
    def __init__(
        self,
        msa_model_name: str = "esm_msa1b_t12_100M_UR50S",
        msa_model_path: Optional[str] = None,
        num_classes: int = 4,
        hidden_dim_multiplier: int = 2,
        dropout: float = 0.1,
        use_cuda: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize MSA Transformer baseline classifier.
        
        Args:
            msa_model_name: Name of MSA Transformer model to use
            msa_model_path: Path to local MSA model weights (if any)
            num_classes: Number of output classes
            hidden_dim_multiplier: Multiplier for hidden dimension in MLP
            dropout: Dropout rate for MLP
            use_cuda: Whether to use CUDA for MSA model
            device: Device to run model on
        """
        super(MSATransformerClassifier, self).__init__()
        
        self._device = device
        self.num_classes = num_classes
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading MSA Transformer model: {msa_model_name}")
        
        # Load MSA Transformer model using EvoMSATransformer wrapper
        try:
            self.msa_model = EvoMSATransformer(
                model_path=msa_model_name if msa_model_path is None else msa_model_path,
                use_cuda=use_cuda,
                device=device
            )
            self.logger.info("Successfully loaded MSA Transformer model")
        except Exception as e:
            self.logger.error(f"Failed to load MSA Transformer model: {e}")
            raise
        
        # Freeze MSA Transformer backbone
        for param in self.msa_model.parameters():
            param.requires_grad = False
        self.logger.info("Frozen all MSA Transformer parameters")
        
        # Get MSA embedding dimension
        self.msa_embed_dim = self.msa_model.msa_out_dim  # Should be 768 for MSA1B
        self.logger.info(f"MSA embedding dimension: {self.msa_embed_dim}")
        
        # Build MLP classifier head
        hidden_dim = self.msa_embed_dim * hidden_dim_multiplier
        self.classifier = nn.Sequential(
            nn.Linear(self.msa_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Move classifier to device (MSA model handles its own device placement)
        self.classifier.to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Frozen parameters: {total_params - trainable_params:,}")



    def forward(
        self,
        msa_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        **kwargs  # Accept additional unused arguments for compatibility
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MSA Transformer + MLP classifier.
        
        Args:
            msa_tokens: MSA tokens [batch_size, num_seqs, seq_len]
            valid_mask: Mask for valid (non-special) tokens [batch_size, seq_len]
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Dictionary containing:
                - logits: Classification logits [batch_size, seq_len, num_classes]
                - valid_mask: Input valid mask [batch_size, seq_len]
                - msa_features: MSA representations [batch_size, seq_len, msa_embed_dim]
        """
        # Extract MSA Transformer features (frozen)
        with torch.no_grad():
            msa_features, _, _ = self.msa_model(msa_tokens)
            # Shape: [batch_size, seq_len, msa_embed_dim]
            # Note: EvoMSATransformer returns query sequence features
        
        # Apply MLP classifier, Shape: [batch_size, seq_len, num_classes]
        logits = self.classifier(msa_features)
        
        return {
            "logits": logits,
            "valid_mask": valid_mask,
            "msa_features": msa_features,
            "predictions": torch.softmax(logits, dim=-1)
        }



class EvoSiteBeta(nn.Module):
    """
    EvoSite model (Beta version), Sequence + Structure + Evolutionary feature.
    """
    
    def __init__(
        self,
        # ESM model configuration
        esm_model_name: str = "ESM-2-650M",
        
        # ConditionNet configuration
        use_condition_net: bool = True,
        condition_hidden_dim: int = 256,
        condition_attention_heads: int = 8,
        use_relative_pos_in_synergy: bool = True,

        # Structure branch configuration
        use_structure_branch: bool = True,
        structure_hidden_dims: List[int] = [512, 512, 512, 512, 512, 512],  # structure_hidden_dims[-1] is output dimension
        num_relation: int = 7,
        edge_input_dim: Optional[int] = None,
        num_angle_bin: Optional[int] = None,
        structure_batch_norm: bool = True,
        short_cut: bool = True,
        readout: str = "sum",
        
        # Fusion method
        fusion_method: str = "cross_attention",  # "cross_attention" or "concat"

        # Cross-attention configuration
        cross_attention_heads: int = 8,
        cross_attention_layers: int = 3,
        positional_number: int = 0,   # Not used, set to 0 to disable relative position feature
        cross_attention_h_rate: float = 1,     
        cross_attention_dropout: float = 0.1,
        
        # Output configuration
        classification_output_dim: int = 4,
        dropout: float = 0.1,
        
        # General parameters
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_graph_construction: bool = True
    ):

        super(EvoSiteBeta, self).__init__()
        self._device = device
        
        self.use_condition_net = use_condition_net
        self.use_structure_branch = use_structure_branch
        self.fusion_method = fusion_method
        self.use_graph_construction = use_graph_construction

        # Initialize the graph construction module if enabled.
        if self.use_graph_construction:
            self.graph_construction_model = layers.geometry.GraphConstruction(
                node_layers=[layers.geometry.AlphaCarbonNode()],
                edge_layers=[
                    layers.geometry.SequentialEdge(max_distance=2),
                    layers.geometry.SpatialEdge(radius=10, min_distance=5),
                    layers.geometry.KNNEdge(k=10, max_distance=0),              # 这里layers.geometry.KNNEdge(k=10, max_distance=0)的设置可能有问题
                ],
                edge_feature="gearnet"
            )
        else:
            self.graph_construction_model = None

        # Initialize ESM model
        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=esm_model_name)
        self.esm_output_dim = self.esm_model.output_dim
        
        # ConditionNet for regulating sequence features with evolutionary information
        if use_condition_net:
            self.condition_network = ConditionNet2(
                fusion_dim=self.esm_output_dim,
                output_dim=None,        # None means use fusion_dim
                dropout=dropout,
                synergy_hidden_dim=condition_hidden_dim,
                synergy_attention_heads=condition_attention_heads,
                use_residual=True,
                use_relative_pos_in_synergy=use_relative_pos_in_synergy
            )
        
        # Structure branch
        if use_structure_branch:
            self.structure_model = StructureEncoder(
                input_dim=self.esm_output_dim,
                hidden_dims=structure_hidden_dims,
                num_relation=num_relation,
                edge_input_dim=edge_input_dim,
                num_angle_bin=num_angle_bin,
                batch_norm=structure_batch_norm,
                short_cut=short_cut,
                readout=readout,
            )

            if self.fusion_method == "cross_attention":
                assert self.esm_output_dim == structure_hidden_dims[-1], "For cross_attention fusion, ESM output dimension must match structure output dimension"
                self.cross_attention = GlobalMultiHeadAttention(
                    d_model=self.esm_output_dim,
                    heads=cross_attention_heads,
                    n_layers=cross_attention_layers,
                    positional_number=positional_number,  # Set to 0 to disable relative position feature, as we don't provide 'rpm'
                    cross_attn_h_rate=cross_attention_h_rate,
                    dropout=cross_attention_dropout
                )
                classification_input_dim = self.esm_output_dim

            elif self.fusion_method == "concat":
                self.cross_attention = None
                classification_input_dim = self.esm_output_dim + structure_hidden_dims[-1]

        else:
            self.structure_model = None
            self.cross_attention = None
            classification_input_dim = self.esm_output_dim


        self.prediction_head = Predictior(classification_input_dim, classification_output_dim)
        
        # Move model to device
        self.to(device)
    


    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - logits (torch.Tensor): The output logits for each valid residue in the batch.
                                        Shape: (|V_res|, num_classes), where |V_res| is the total number
                                        of residues in the batch.
                - protein_mask (torch.Tensor): A boolean mask indicating the valid residues for each
                                            protein in a padded format. 1 indicates valid residues, and 0 indicates padding.
                                            Shape: (batch_size, max_num_residues). 
        """
        graph = batch["protein_graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        output_esm = self.esm_model(graph, graph.node_feature.float())        # Shape: (|V_res|, esm_output_dim)
        node_feature_seq = output_esm.get("residue_feature")

        protein_mask = None
        
        if self.use_condition_net:
            packed_seq_feats, protein_mask = pack_residue_feats(graph, node_feature_seq)
            padded_conservation_scores, _ = pack_residue_feats(graph, batch["conservation_scores"])
            conditioned_embeddings_packed = self.condition_network(
                fused_embeddings=packed_seq_feats,
                conservation_scores=padded_conservation_scores,          # Pass the correctly shaped 2D tensor
                dca_single_potentials=batch["single_site_potentials"],  # Expected: List of (valid_len, 20)
                coupling_matrix=batch["coupling_matrices"],         # Expected: List of (valid_len, valid_len)
                valid_tokens_mask=protein_mask.bool(),
            )
            
            # node_feature_conditioned Shape: (|V_res|, esm_output_dim)
            node_feature_conditioned = conditioned_embeddings_packed[protein_mask.bool()]

        # Structure Feature Extraction
        if self.use_structure_branch:
            output_struct = self.structure_model(graph, node_feature_seq)
            node_feature_struct = output_struct.get("node_feature", output_struct.get("residue_feature"))   # node_feature_struct Shape: (|V_res|, structure_output_dim)
            
            # Multi-modal Feature Fusion
            if self.fusion_method == "concat":
                # final_node_feature Shape: (|V_res|, esm_output_dim + structure_output_dim)
                final_node_feature = torch.cat([node_feature_seq, node_feature_struct], dim=-1) if not self.use_condition_net else torch.cat([node_feature_conditioned, node_feature_struct], dim=-1)
            
            elif self.fusion_method == "cross_attention":
                packed_seq_feats, protein_mask = pack_residue_feats(graph, node_feature_seq) if not self.use_condition_net else pack_residue_feats(graph, node_feature_conditioned)
                packed_struct_feats, _ = pack_residue_feats(graph, node_feature_struct)
                
                fused_packed_features, _, _, _, _ = self.cross_attention(
                    src=packed_seq_feats,
                    tgt=packed_struct_feats,
                    rpm=None,  # No relative position matrix available in this pipeline
                    src_mask=protein_mask.bool(),
                    tgt_mask=protein_mask.bool()
                )

                # Convert back to packed format.
                # final_node_feature Shape: (|V_res|, esm_output_dim)
                final_node_feature = fused_packed_features[protein_mask.bool()]
        else:
            # If no structure branch is used, the final features are the (possibly conditioned) sequence features.
            final_node_feature = node_feature_seq if not self.use_condition_net else node_feature_conditioned

        # Ensure the mask is computed if it wasn't needed before.
        if protein_mask is None:
            _, protein_mask = pack_residue_feats(graph, final_node_feature)

        # logits Shape: (|V_res|, num_classes)
        logits = self.prediction_head(final_node_feature)
        
        return logits, protein_mask



class EvoSiteGamma(EvoSiteBeta):
    """
    https://aistudio.google.com/prompts/1CtNs_pXAsrm5Giw7p4ZrRBXn61S2kiMp
    """
    
    def __init__(
        self,
        # ESM model configuration
        esm_model_name: str = "ESM-2-650M",

        # Structure branch configuration (mandatory for Gamma)
        structure_hidden_dims: List[int] = [512, 512, 512, 512, 512, 512],
        num_relation: int = 7,
        edge_input_dim: Optional[int] = None,
        num_angle_bin: Optional[int] = None,
        structure_batch_norm: bool = True,
        short_cut: bool = True,
        readout: str = "sum",
        
        # Output configuration
        classification_output_dim: int = 4,
        dropout: float = 0.1,
        
        # General parameters
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_graph_construction: bool = True,
        
        # --- Control flags for all four auxiliary tasks ---
        use_aux_coevolution_head: bool = True,
        use_aux_distance_head: bool = True,
        use_conservation_weighted_mlm: bool = False,
        use_coevolutionary_reconstruction: bool = False,
        
        # Auxiliary head parameters
        aux_head_hidden_dim: int = 128,
        num_distance_bins: int = 20,

        # MLM-specific parameters
        mlm_mask_token_id: int = 32, # Default mask token ID for ESM-2 alphabet

        **kwargs
    ):
        # Bypass EvoSiteBeta's __init__ to redefine the architecture
        nn.Module.__init__(self)
        self._device = device
        
        # Store configurations for all auxiliary tasks
        self.use_graph_construction = use_graph_construction
        self.use_aux_coevolution_head = use_aux_coevolution_head
        self.use_aux_distance_head = use_aux_distance_head
        self.use_conservation_weighted_mlm = use_conservation_weighted_mlm
        self.use_coevolutionary_reconstruction = use_coevolutionary_reconstruction
        
        # --- Initialize Backbone Components (similar to EvoSiteBeta) ---
        if self.use_graph_construction:
            self.graph_construction_model = layers.geometry.GraphConstruction(
                node_layers=[layers.geometry.AlphaCarbonNode()],
                edge_layers=[
                    layers.geometry.SequentialEdge(max_distance=2),
                    layers.geometry.SpatialEdge(radius=10, min_distance=5),
                    layers.geometry.KNNEdge(k=10, max_distance=0),
                ],
                edge_feature="gearnet"
            )
        else:
            self.graph_construction_model = None

        # Main ESM model for sequence and structure feature extraction (no LM head)
        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=esm_model_name)
        self.esm_output_dim = self.esm_model.output_dim
        
        # Structure branch for geometric feature extraction
        self.structure_model = StructureEncoder(
            input_dim=self.esm_output_dim,
            hidden_dims=structure_hidden_dims,
            num_relation=num_relation,
            edge_input_dim=edge_input_dim,
            num_angle_bin=num_angle_bin,
            batch_norm=structure_batch_norm,
            short_cut=short_cut,
            readout=readout,
        )

        # Fusion is fixed to concatenation for all prediction heads
        fused_feature_dim = self.esm_output_dim + structure_hidden_dims[-1]
        
        # --- Initialize Prediction Heads ---
        # 1. Main Head for Active Site Classification
        self.prediction_head = Predictior(fused_feature_dim, classification_output_dim)

        # 2. Auxiliary Head for Co-evolution Prediction
        if self.use_aux_coevolution_head:
            self.coevolution_head = CoevolutionPredictionHead(
                input_dim=fused_feature_dim, hidden_dim=aux_head_hidden_dim, dropout=dropout
            )

        # 3. Auxiliary Head for Distance Prediction
        if self.use_aux_distance_head:
            self.distance_head = DistancePredictionHead(
                input_dim=fused_feature_dim, hidden_dim=aux_head_hidden_dim, num_bins=num_distance_bins, dropout=dropout
            )
            
        # 4. Separate ESM model with LM head for MLM-based auxiliary tasks
        self.esm_model_for_mlm = None
        if self.use_conservation_weighted_mlm or self.use_coevolutionary_reconstruction:
            # Load a standard ESM-2 model with its language model head enabled
            self.esm_model_for_mlm, _ = esm.pretrained.load_model_and_alphabet(esm_model_name)
            # Freeze the MLM model to use it only as a feature extractor/logit generator
            for param in self.esm_model_for_mlm.parameters():
                param.requires_grad = False
            self.esm_model_for_mlm = self.esm_model_for_mlm.to(device)
            logging.info(f"Initialized a separate ESM model with LM head for MLM tasks.")

        self.to(device)


    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for EvoSiteGamma with multi-task outputs.

        Returns:
            A dictionary containing predictions for all active tasks.
        """
        # --- 1. Backbone Feature Extraction & Fusion ---
        graph = batch["protein_graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        output_esm = self.esm_model(graph, graph.node_feature.float())
        node_feature_seq = output_esm.get("residue_feature")

        output_struct = self.structure_model(graph, node_feature_seq)
        node_feature_struct = output_struct.get("node_feature", output_struct.get("residue_feature"))
        
        # Fusion by concatenation
        final_node_feature = torch.cat([node_feature_seq, node_feature_struct], dim=-1)

        _, protein_mask = pack_residue_feats(graph, final_node_feature)

        # --- 2. Main Prediction Task ---
        logits = self.prediction_head(final_node_feature)
        
        outputs = {"logits": logits, "protein_mask": protein_mask}

        # --- 3. Auxiliary Tasks ---
        # Pairwise prediction tasks (Co-evolution and Distance)
        if self.training and (self.use_aux_coevolution_head or self.use_aux_distance_head):
            num_residues = final_node_feature.size(0)
            feat_i = final_node_feature.unsqueeze(1).expand(-1, num_residues, -1)
            feat_j = final_node_feature.unsqueeze(0).expand(num_residues, -1, -1)
            pair_features = torch.cat([feat_i, feat_j], dim=-1)

            if self.use_aux_coevolution_head:
                outputs["predicted_couplings"] = self.coevolution_head(pair_features)

            if self.use_aux_distance_head:
                outputs["predicted_distance_logits"] = self.distance_head(pair_features)
        
        # MLM-based tasks (Conservation-weighted MLM and Co-evolutionary Reconstruction)
        if self.training and self.esm_model_for_mlm is not None:
            # Check for masked tokens prepared by the trainer
            if 'mlm_input_tokens' in batch:
                mlm_tokens = batch['mlm_input_tokens']
                # Get logits from the ESM model with LM head
                with torch.no_grad(): # The MLM model is frozen
                    mlm_results = self.esm_model_for_mlm(mlm_tokens, repr_layers=[self.esm_model_for_mlm.num_layers])
                outputs['lm_head_logits'] = mlm_results['logits']

        return outputs



class EvoSitePhiGnet(nn.Module):
    """
    EvoSite model (PhiGnet version)

    https://aistudio.google.com/prompts/1FeDYpDr-bTvH2inWHxbT52y4yb3jOqv8
    """
    def __init__(
        self,
        # ESM model configuration
        esm_model_name: str = "ESM-2-650M",

        # Structure branch configuration
        structure_hidden_dims: List[int] = [512, 512, 512, 512, 512, 512],
        num_relation: int = 7,
        edge_input_dim: Optional[int] = None,
        num_angle_bin: Optional[int] = None,
        structure_batch_norm: bool = True,
        short_cut: bool = True,
        readout: str = "sum",

        # PhiGnet GCN configuration
        phignet_gcn_hidden_dim: int = 256,  # Hidden dim for each of the 3 GCN layers
        phignet_gcn_layers: int = 3,       # Number of GCN layers, should be 3 to match PhiGnet
        phignet_normalization: bool = False,
        phignet_cut_thresh: float = 0.2,
        phignet_binary_weights: bool = False,

        # Output configuration
        classification_output_dim: int = 4,

        # General parameters
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_graph_construction: bool = True,
    ):
        super().__init__()
        self._device = device
        self.use_graph_construction = use_graph_construction

        # Initialize the graph construction module if enabled.
        if self.use_graph_construction:
            self.graph_construction_model = layers.geometry.GraphConstruction(
                node_layers=[layers.geometry.AlphaCarbonNode()],
                edge_layers=[
                    layers.geometry.SequentialEdge(max_distance=2),
                    layers.geometry.SpatialEdge(radius=10, min_distance=5),
                    layers.geometry.KNNEdge(k=10, max_distance=0),
                ],
                edge_feature="gearnet"
            )
        else:
            self.graph_construction_model = None


        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=esm_model_name)
        self.esm_output_dim = self.esm_model.output_dim
        

        self.structure_model = StructureEncoder(
            input_dim=self.esm_output_dim,
            hidden_dims=structure_hidden_dims,
            num_relation=num_relation,
            edge_input_dim=edge_input_dim,
            num_angle_bin=num_angle_bin,
            batch_norm=structure_batch_norm,
            short_cut=short_cut,
            readout=readout,
        )
        structure_output_dim = structure_hidden_dims[-1]


        # Initialize Sparse PhiGnet Branch Components
        self.coupling_processor = PhiGnetCouplingProcessor(
            cut_thresh=phignet_cut_thresh,
            use_binary_weights=phignet_binary_weights,
            normalize_before_threshold=phignet_normalization
        )
        self.graph_conv = PhiGnetGraphConv(
            input_dim=self.esm_output_dim,
            hidden_dim=phignet_gcn_hidden_dim,
            num_layers=phignet_gcn_layers,
            use_bias=False # PhiGnet's GraphConv layers had use_bias=False
        )
        phignet_output_dim = phignet_gcn_hidden_dim * phignet_gcn_layers
        

        # Initialize Final Prediction Head
        self.prediction_head = Predictior(self.esm_output_dim + structure_output_dim + phignet_output_dim, classification_output_dim)
        
        self.to(device)



    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        graph = batch["protein_graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        output_esm = self.esm_model(graph, graph.node_feature.float())
        node_feature_seq = output_esm.get("residue_feature")

        output_struct = self.structure_model(graph, node_feature_seq)
        node_feature_struct = output_struct.get("node_feature")

        gcn_input_feats = node_feature_seq
        coupling_matrices = batch["coupling_matrices"]
        edge_index, edge_weight, batch_index = self.coupling_processor(coupling_matrices, graph.num_residues)
        node_feature_phignet = self.graph_conv(gcn_input_feats, edge_index, edge_weight)

        # Feature Fusion and Prediction
        final_node_feature = torch.cat([node_feature_seq, node_feature_struct, node_feature_phignet], dim=-1)
        logits = self.prediction_head(final_node_feature) # Shape: (|V_res|, num_classes)
        
        # Reconstruct the protein mask for the trainer's validation logic
        batch_size = (batch_index.max().item() + 1) if batch_index.numel() > 0 else 0
        if batch_size > 0:
            num_nodes_per_graph = torch.bincount(batch_index, minlength=batch_size)
            max_len_in_batch = num_nodes_per_graph.max().item()
            protein_mask = torch.arange(max_len_in_batch, device=self._device)[None, :] < num_nodes_per_graph[:, None]
        else:
            raise ValueError("Batch index tensor is empty, cannot determine batch size.")

        return logits, protein_mask



class EvoSiteBetaPhiGnet(nn.Module):
    """
    对话：https://aistudio.google.com/prompts/1qo9zWcAUcqIO1IB-8uXorfsQV-pQ0iWy
    """
    def __init__(
        self,
        # ESM model configuration
        esm_model_name: str = "ESM-2-650M",

        # ConditionNet3 configuration
        use_condition_net: bool = True,
        condition_hidden_dim: int = 256,
        condition_attention_heads: int = 8,
        use_relative_pos_in_synergy: bool = True,
        condition_coupling_norm: bool = False,
        condition_coupling_binary: bool = False,
        condition_coupling_thresh: float = 0.0,

        # Structure branch configuration
        structure_hidden_dims: List[int] = [512, 512, 512, 512, 512, 512],
        num_relation: int = 7,
        edge_input_dim: Optional[int] = None,
        num_angle_bin: Optional[int] = None,
        structure_batch_norm: bool = True,
        short_cut: bool = True,
        readout: str = "sum",

        # PhiGnet GCN configuration
        phignet_gcn_hidden_dim: int = 256,
        phignet_gcn_layers: int = 3,
        phignet_normalization: bool = False,
        phignet_cut_thresh: float = 0.2,
        phignet_binary_weights: bool = False,

        # Output configuration
        classification_output_dim: int = 4,
        dropout: float = 0.1,

        # General parameters
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_graph_construction: bool = True,
    ):
        super().__init__()
        self._device = device
        self.use_graph_construction = use_graph_construction
        self.use_condition_net = use_condition_net

        # Initialize the graph construction module if enabled.
        if self.use_graph_construction:
            self.graph_construction_model = layers.geometry.GraphConstruction(
                node_layers=[layers.geometry.AlphaCarbonNode()],
                edge_layers=[
                    layers.geometry.SequentialEdge(max_distance=2),
                    layers.geometry.SpatialEdge(radius=10, min_distance=5),
                    layers.geometry.KNNEdge(k=10, max_distance=0),
                ],
                edge_feature="gearnet"
            )
        else:
            self.graph_construction_model = None

        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=esm_model_name)
        self.esm_output_dim = self.esm_model.output_dim
        
        if use_condition_net:
            self.condition_network = ConditionNet3(
                fusion_dim=self.esm_output_dim,
                output_dim=None,  # Use fusion_dim as output_dim
                dropout=dropout,
                synergy_hidden_dim=condition_hidden_dim,
                synergy_attention_heads=condition_attention_heads,
                use_residual=True,
                use_relative_pos_in_synergy=use_relative_pos_in_synergy,
                use_coupling_norm=condition_coupling_norm,
                use_coupling_binary=condition_coupling_binary,
                coupling_thresh=condition_coupling_thresh,
            )

        self.structure_model = StructureEncoder(
            input_dim=self.esm_output_dim,
            hidden_dims=structure_hidden_dims,
            num_relation=num_relation,
            edge_input_dim=edge_input_dim,
            num_angle_bin=num_angle_bin,
            batch_norm=structure_batch_norm,
            short_cut=short_cut,
            readout=readout,
        )
        structure_output_dim = structure_hidden_dims[-1]


        # Initialize Sparse PhiGnet Branch Components
        self.coupling_processor = PhiGnetCouplingProcessor(
            cut_thresh=phignet_cut_thresh,
            use_binary_weights=phignet_binary_weights,
            normalize_before_threshold=phignet_normalization
        )
        self.graph_conv = PhiGnetGraphConv(
            input_dim=self.esm_output_dim,
            hidden_dim=phignet_gcn_hidden_dim,
            num_layers=phignet_gcn_layers,
            use_bias=False
        )
        phignet_output_dim = phignet_gcn_hidden_dim * phignet_gcn_layers


        # Initialize Final Prediction Head
        self.prediction_head = Predictior(self.esm_output_dim + structure_output_dim + phignet_output_dim, classification_output_dim)
        
        self.to(device)



    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        graph = batch["protein_graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        output_esm = self.esm_model(graph, graph.node_feature.float())
        node_feature_seq = output_esm.get("residue_feature")


        if self.use_condition_net:
            packed_seq_feats, protein_mask = pack_residue_feats(graph, node_feature_seq)
            padded_conservation_scores, _ = pack_residue_feats(graph, batch["conservation_scores"])
            
            conditioned_embeddings_packed = self.condition_network(
                fused_embeddings=packed_seq_feats,
                conservation_scores=padded_conservation_scores,
                dca_single_potentials=batch["single_site_potentials"],
                coupling_matrix=batch["coupling_matrices"],
                valid_tokens_mask=protein_mask.bool(),
            )
            node_feature_conditioned = conditioned_embeddings_packed[protein_mask.bool()]


        output_struct = self.structure_model(graph, node_feature_seq)
        node_feature_struct = output_struct.get("node_feature")


        gcn_input_feats = node_feature_seq
        coupling_matrices = batch["coupling_matrices"]
        edge_index, edge_weight, batch_index = self.coupling_processor(coupling_matrices, graph.num_residues)
        node_feature_phignet = self.graph_conv(gcn_input_feats, edge_index, edge_weight)

        # Feature Fusion and Prediction
        final_node_feature = torch.cat([node_feature_seq, node_feature_struct, node_feature_phignet], dim=-1) if not self.use_condition_net else torch.cat([node_feature_conditioned, node_feature_struct, node_feature_phignet], dim=-1)
        logits = self.prediction_head(final_node_feature) # Shape: (|V_res|, num_classes)
        
        # Reconstruct the protein mask for the trainer's validation logic
        batch_size = (batch_index.max().item() + 1) if batch_index.numel() > 0 else 0
        if batch_size > 0:
            num_nodes_per_graph = torch.bincount(batch_index, minlength=batch_size)
            max_len_in_batch = num_nodes_per_graph.max().item()
            protein_mask = torch.arange(max_len_in_batch, device=self._device)[None, :] < num_nodes_per_graph[:, None]
        else:
            raise ValueError("Batch index tensor is empty, cannot determine batch size.")

        return logits, protein_mask



class EvoSiteGraphEC(nn.Module):
    def __init__(
        self,
        # ESM model configuration
        esm_model_name: str = "ESM-2-650M",
        
        # ConditionNet configuration
        condition_net_version: Optional[str] = "v2", # None, "v1", "v2", "v3"
        condition_hidden_dim: int = 256,
        condition_attention_heads: int = 8,
        use_relative_pos_in_synergy: bool = True,
        condition_coupling_norm: bool = False,          # Only for condition_net_version == "v3"
        condition_coupling_binary: bool = False,        # Only for condition_net_version == "v3"
        condition_coupling_thresh: float = 0.0,         # Only for condition_net_version == "v3"

        # Structure branch configuration
        structure_hidden_dims: List[int] = [512] * 6,
        num_relation: int = 7,
        edge_input_dim: Optional[int] = None,
        num_angle_bin: Optional[int] = None,
        structure_batch_norm: bool = True,
        short_cut: bool = True,
        readout: str = "sum",
        
        # GraphEC branch configuration
        use_graphec: bool = True,
        graphec_hidden_dim: int = 128,
        graphec_output_dim: int = 128,                  # Must be the same as base_fused_dim if using 'cross_attention' fusion
        graphec_layers: int = 4,
        graphec_dropout: float = 0.2,
        dssp_dim: int = 9,
        graphec_node_feature_dim: int = 184,
        graphec_edge_feature_dim: int = 450,

        # Fusion method
        graphec_fusion_method: Literal["concat", "base_to_graphec", "graphec_to_base", "bidirectional"] = "concat",
        bidirectional_fusion_mode: Literal["concat", "add"] = "add",
        
        # Cross-attention configuration
        cross_attention_heads: int = 8,
        cross_attention_layers: int = 3,
        positional_number: int = 0,
        cross_attention_h_rate: float = 1,  
        cross_attention_dropout: float = 0.1,

        # Output configuration
        classification_output_dim: int = 4,
        dropout: float = 0.1,
        
        # General parameters
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_graph_construction: bool = True
    ):
        super(EvoSiteGraphEC, self).__init__()
        self._device = device
        self.dssp_dim = dssp_dim

        self.use_graphec = use_graphec
        self.condition_net_version = condition_net_version
        self.graphec_fusion_method = graphec_fusion_method
        self.bidirectional_fusion_mode = bidirectional_fusion_mode
        self.use_graph_construction = use_graph_construction

        # --- Graph Construction ---
        if self.use_graph_construction:
            self.graph_construction_model = layers.geometry.GraphConstruction(
                node_layers=[layers.geometry.AlphaCarbonNode()],
                edge_layers=[
                    layers.geometry.SequentialEdge(max_distance=2),
                    layers.geometry.SpatialEdge(radius=10, min_distance=5),
                    layers.geometry.KNNEdge(k=10, max_distance=0),
                ],
                edge_feature="gearnet"
            )
        else:
            self.graph_construction_model = None

        # --- Branch 1: ESM + Co-evolution Features ---
        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=esm_model_name)
        self.esm_output_dim = self.esm_model.output_dim
        
        self.condition_network = None
        if self.condition_net_version:
            if self.condition_net_version == "v1":      # 暂未检查
                self.condition_network = ConditionNet(fusion_dim=self.esm_output_dim, synergy_hidden_dim=condition_hidden_dim)
            elif self.condition_net_version == "v2":
                self.condition_network = ConditionNet2(
                    fusion_dim=self.esm_output_dim,
                    output_dim=None,
                    dropout=dropout,
                    synergy_hidden_dim=condition_hidden_dim,
                    synergy_attention_heads=condition_attention_heads,
                    use_residual=True,
                    use_relative_pos_in_synergy=use_relative_pos_in_synergy
                )
            elif self.condition_net_version == "v3":
                self.condition_network = ConditionNet3(
                    fusion_dim=self.esm_output_dim,
                    output_dim=None,
                    dropout=dropout,
                    synergy_hidden_dim=condition_hidden_dim,
                    synergy_attention_heads=condition_attention_heads,
                    use_residual=True,
                    use_relative_pos_in_synergy=use_relative_pos_in_synergy,
                    use_coupling_norm=condition_coupling_norm,
                    use_coupling_binary=condition_coupling_binary,
                    coupling_thresh=condition_coupling_thresh,
                )
            else:
                raise ValueError(f"Unsupported condition_net_version: {self.condition_net_version}")

        # --- Branch 2: Structure Features (GearNet) ---
        self.structure_model = StructureEncoder(
            input_dim=self.esm_output_dim,
            hidden_dims=structure_hidden_dims,
            num_relation=num_relation,
            edge_input_dim=edge_input_dim,
            num_angle_bin=num_angle_bin,
            batch_norm=structure_batch_norm,
            short_cut=short_cut,
            readout=readout,
        )
        structure_output_dim = structure_hidden_dims[-1]
        base_fused_dim = self.esm_output_dim + structure_output_dim

        # --- Branch 3: GraphEC GNN Features ---
        self.feature_calculator = GraphECFeatureCalculator()

        if self.use_graphec:
            graphec_node_input_dim = self.esm_output_dim + dssp_dim + graphec_node_feature_dim
            self.graphec_encoder = GraphECEncoder(
                node_in_dim=graphec_node_input_dim,
                edge_in_dim=graphec_edge_feature_dim,
                hidden_dim=graphec_hidden_dim,
                output_dim=graphec_output_dim,
                num_layers=graphec_layers,
                drop_rate=graphec_dropout
            )
        
        # --- Fusion and Final Prediction Head ---
        self.attn_base_to_graphec = None
        self.attn_graphec_to_base = None
        if self.use_graphec:
            if "attention" in self.graphec_fusion_method or "base" in self.graphec_fusion_method or "bidirectional" in self.graphec_fusion_method:
                assert base_fused_dim == graphec_output_dim, f"For any attention-based fusion, base_fused_dim ({base_fused_dim}) must match graphec_output_dim ({graphec_output_dim})."

            if self.graphec_fusion_method == "concat":
                classification_input_dim = base_fused_dim + graphec_output_dim
            
            elif self.graphec_fusion_method == "base_to_graphec":
                self.attn_base_to_graphec = GlobalMultiHeadAttention(
                    d_model=base_fused_dim, heads=cross_attention_heads, n_layers=cross_attention_layers,
                    positional_number=positional_number, cross_attn_h_rate=cross_attention_h_rate, dropout=cross_attention_dropout
                )
                classification_input_dim = base_fused_dim

            elif self.graphec_fusion_method == "graphec_to_base":
                self.attn_graphec_to_base = GlobalMultiHeadAttention(
                    d_model=base_fused_dim, heads=cross_attention_heads, n_layers=cross_attention_layers,
                    positional_number=positional_number, cross_attn_h_rate=cross_attention_h_rate, dropout=cross_attention_dropout
                )
                classification_input_dim = base_fused_dim

            elif self.graphec_fusion_method == "bidirectional":
                self.attn_base_to_graphec = GlobalMultiHeadAttention(
                    d_model=base_fused_dim, heads=cross_attention_heads, n_layers=cross_attention_layers,
                    positional_number=positional_number, cross_attn_h_rate=cross_attention_h_rate, dropout=cross_attention_dropout
                )
                self.attn_graphec_to_base = GlobalMultiHeadAttention(
                    d_model=base_fused_dim, heads=cross_attention_heads, n_layers=cross_attention_layers,
                    positional_number=positional_number, cross_attn_h_rate=cross_attention_h_rate, dropout=cross_attention_dropout
                )
                if self.bidirectional_fusion_mode == "concat":
                    classification_input_dim = base_fused_dim * 2
                elif self.bidirectional_fusion_mode == "add":
                    classification_input_dim = base_fused_dim
                else:
                    raise ValueError(f"Unsupported bidirectional_fusion_mode: {self.bidirectional_fusion_mode}")
            else:
                raise ValueError(f"Unsupported graphec_fusion_method: {self.graphec_fusion_method}")
        else:
            classification_input_dim = base_fused_dim
            
        self.prediction_head = Predictior(classification_input_dim, classification_output_dim, dropout=dropout)
        self.to(device)



    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        graph = batch["protein_graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        output_esm = self.esm_model(graph, graph.node_feature.float())
        node_feature_seq = output_esm.get("residue_feature")
        
        if self.condition_network:
            packed_seq_feats, protein_mask = pack_residue_feats(graph, node_feature_seq)
            
            if self.condition_net_version == "v1":      # 暂未检查
                # ConditionNet v1 requires valid_residue_indices, which we generate from the mask.
                valid_residue_indices = [mask.nonzero().squeeze(-1) for mask in protein_mask]
                
                # It also expects padded conservation scores.
                padded_conservation_scores, _ = pack_residue_feats(graph, batch["conservation_scores"])

                conditioned_embeddings_packed = self.condition_network(
                    fused_embeddings=packed_seq_feats,
                    conservation_scores=padded_conservation_scores,
                    dca_single_potentials=batch["single_site_potentials"], # Pass the list
                    coupling_matrix=batch["coupling_matrices"],         # Pass the list
                    valid_residue_indices=valid_residue_indices,        # Pass the generated indices
                    valid_tokens_mask=protein_mask.bool(),
                )
            elif self.condition_net_version in ["v2", "v3"]:
                padded_conservation_scores, _ = pack_residue_feats(graph, batch["conservation_scores"])
                conditioned_embeddings_packed = self.condition_network(
                    fused_embeddings=packed_seq_feats,
                    conservation_scores=padded_conservation_scores,
                    dca_single_potentials=batch["single_site_potentials"],
                    coupling_matrix=batch["coupling_matrices"],
                    valid_tokens_mask=protein_mask.bool(),
                )
            
            # node_feature_conditioned Shape: (|V_res|, esm_output_dim)
            node_feature_conditioned = conditioned_embeddings_packed[protein_mask.bool()]
        else:
            node_feature_conditioned = node_feature_seq
            _, protein_mask = pack_residue_feats(graph, node_feature_seq)

        output_struct = self.structure_model(graph, node_feature_seq)
        node_feature_struct = output_struct.get("node_feature")

        base_fused_feature = torch.cat([node_feature_conditioned, node_feature_struct], dim=-1)

        # --- GraphEC Branch ---
        if self.use_graphec:

            graphec_node_feature, graphec_edge_feature = self.feature_calculator(
                graph=graph,
                atom_coords=batch["atom_coords"],
                edge_index=batch["batched_graphec_edge_indices"]
            )

            assert batch["dssp_features"].shape[-1] == self.dssp_dim, f"DSSP feature dimension mismatch! Expected {self.dssp_dim}, but got {batch['dssp_features'].shape[-1]}."
            graphec_h_v = torch.cat([
                node_feature_seq,
                batch["dssp_features"], 
                graphec_node_feature
            ], dim=-1)
            
            graphec_output_feature = self.graphec_encoder(
                h_V=graphec_h_v,
                edge_index=batch["batched_graphec_edge_indices"],
                h_E=graphec_edge_feature,
                batch_id=batch["graphec_batch_id"]
            )

            # --- Final Fusion ---
            if self.graphec_fusion_method == "concat":
                final_node_feature = torch.cat([base_fused_feature, graphec_output_feature], dim=-1)

            elif self.graphec_fusion_method == "base_to_graphec":
                packed_base, _ = pack_residue_feats(graph, base_fused_feature)
                packed_graphec, _ = pack_residue_feats(graph, graphec_output_feature)
                fused_packed, _, _, _, _ = self.attn_base_to_graphec(
                    src=packed_base, tgt=packed_graphec, rpm=None,
                    src_mask=protein_mask.bool(), tgt_mask=protein_mask.bool()
                )
                final_node_feature = fused_packed[protein_mask.bool()]

            elif self.graphec_fusion_method == "graphec_to_base":
                packed_base, _ = pack_residue_feats(graph, base_fused_feature)
                packed_graphec, _ = pack_residue_feats(graph, graphec_output_feature)
                fused_packed, _, _, _, _ = self.attn_graphec_to_base(
                    src=packed_graphec, tgt=packed_base, rpm=None,
                    src_mask=protein_mask.bool(), tgt_mask=protein_mask.bool()
                )
                final_node_feature = fused_packed[protein_mask.bool()]

            elif self.graphec_fusion_method == "bidirectional":
                packed_base, _ = pack_residue_feats(graph, base_fused_feature)
                packed_graphec, _ = pack_residue_feats(graph, graphec_output_feature)
                
                # First direction: base -> graphec
                fused_base_packed, _, _, _, _ = self.attn_base_to_graphec(
                    src=packed_base, tgt=packed_graphec, rpm=None,
                    src_mask=protein_mask.bool(), tgt_mask=protein_mask.bool()
                )
                
                # Second direction: graphec -> base
                fused_graphec_packed, _, _, _, _ = self.attn_graphec_to_base(
                    src=packed_graphec, tgt=packed_base, rpm=None,
                    src_mask=protein_mask.bool(), tgt_mask=protein_mask.bool()
                )
                
                # Combine the two fused features
                if self.bidirectional_fusion_mode == "concat":
                    combined_packed = torch.cat([fused_base_packed, fused_graphec_packed], dim=-1)
                elif self.bidirectional_fusion_mode == "add":
                    combined_packed = fused_base_packed + fused_graphec_packed
                
                final_node_feature = combined_packed[protein_mask.bool()]
        else:
            final_node_feature = base_fused_feature
            
        logits = self.prediction_head(final_node_feature)
        
        return logits, protein_mask
    


class EvoSiteDelta(nn.Module):
    
    def __init__(
        self,
        # ESM model configuration
        esm_model_name: str = "ESM-2-650M",

        # Structure model configuration
        structure_hidden_dims: List[int] = [512, 512, 512, 512, 512, 512],  # structure_hidden_dims[-1] is output dimension
        num_relation: int = 7,
        edge_input_dim: Optional[int] = None,
        num_angle_bin: Optional[int] = None,
        structure_batch_norm: bool = True,
        short_cut: bool = True,
        readout: str = "sum",
        
        # Reaction model configuration
        rxn_model_path: Optional[str] = None, # If None, the reaction branch is disabled
        from_scratch: bool = False,

        # ConditionNet configuration
        condition_net_version: Optional[str] = "v2", # None, "v1", "v2", "v3"
        condition_hidden_dim: int = 256,
        condition_attention_heads: int = 8,
        use_relative_pos_in_synergy: bool = True,
        condition_coupling_norm: bool = False,          # Only for condition_net_version == "v3"
        condition_coupling_binary: bool = False,        # Only for condition_net_version == "v3"
        condition_coupling_thresh: float = 0.0,         # Only for condition_net_version == "v3"

        # CouplingFeatureEncoder configuration
        use_coupling_encoder: bool = False,         
        coupling_encoder_output_dim: int = 128, 
        coupling_encoder_embed_dim: int = 64, 
        coupling_encoder_num_heads: int = 8, 
        coupling_encoder_pooling: str = 'cls',          # ('cls', 'mean', 'max')
        coupling_encoder_chunk_size: int = 1024,

        # PhiGnet-style GCN configuration
        use_phignet: bool = False,
        phignet_gcn_hidden_dim: int = 256,
        phignet_gcn_layers: int = 3,
        phignet_normalization: bool = False,
        phignet_cut_thresh: float = 0.2,
        phignet_binary_weights: bool = True,

        # DSSP configuration
        use_dssp: bool = False,
        dssp_dim: int = 9,

        # GraphEC configuration
        use_graphec: bool = False,
        graphec_hidden_dim: int = 128,
        graphec_output_dim: int = 256,
        graphec_layers: int = 4,
        graphec_dropout: float = 0.2,
        graphec_node_feature_dim: int = 184,
        graphec_edge_feature_dim: int = 450,
        
        # Enzyme-Reaction Cross-attention configuration
        cross_attention_heads: int = 8,
        cross_attention_layers: int = 3,
        positional_number: int = 0,   # Not used, set to 0 to disable relative position feature
        cross_attention_h_rate: float = 1,     
        cross_attention_dropout: float = 0.1,

        # Output configuration
        classification_output_dim: int = 4,
        dropout: float = 0.1,

        # General configuration
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_graph_construction: bool = True,
    ):

        super(EvoSiteDelta, self).__init__()
        self._device = device
        self.use_graph_construction = use_graph_construction
        self.rxn_model_path = rxn_model_path
        self.condition_net_version = condition_net_version
        self.use_coupling_encoder = use_coupling_encoder
        self.use_phignet = use_phignet
        self.use_dssp = use_dssp
        self.use_graphec = use_graphec

        
        # Graph Construction
        if self.use_graph_construction:
            self.graph_construction_model = layers.geometry.GraphConstruction(
                node_layers=[layers.geometry.AlphaCarbonNode()],
                edge_layers=[
                    layers.geometry.SequentialEdge(max_distance=2),
                    layers.geometry.SpatialEdge(radius=10, min_distance=5),
                    layers.geometry.KNNEdge(k=10, max_distance=0),
                ],
                edge_feature="gearnet"
            )
        else:
            self.graph_construction_model = None


        # Sequence Feature Extractor
        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=esm_model_name)
        self.esm_output_dim = self.esm_model.output_dim


        # Structure Feature Extractor
        self.structure_model = StructureEncoder(
            input_dim=self.esm_output_dim,
            hidden_dims=structure_hidden_dims,
            num_relation=num_relation,
            edge_input_dim=edge_input_dim,
            num_angle_bin=num_angle_bin,
            batch_norm=structure_batch_norm,
            short_cut=short_cut,
            readout=readout,
        )
        structure_output_dim = structure_hidden_dims[-1]
        

        # Reaction Feature Extractor
        if rxn_model_path:
            self.rxn_attn_model = self._load_rxn_attn_model(rxn_model_path, from_scratch)
        else:
            self.rxn_attn_model = None


        # ConditionNet for Evolutionary Feature Modulation
        self.condition_network = None
        if self.condition_net_version == "v2":
            self.condition_network = ConditionNet2(
                fusion_dim=self.esm_output_dim,
                output_dim=None,        # None means use fusion_dim
                dropout=dropout,
                synergy_hidden_dim=condition_hidden_dim,
                synergy_attention_heads=condition_attention_heads,
                use_residual=True,
                use_relative_pos_in_synergy=use_relative_pos_in_synergy
            )
        elif self.condition_net_version is None:
            pass
        else:
            raise ValueError(f"Unsupported condition_net_version: '{self.condition_net_version}'. "
                             f"Valid options include 'v2' or None.")
        

        self.coupling_feature_encoder = None
        coupling_encoder_final_output_dim = 0
        if self.use_coupling_encoder:
            self.coupling_feature_encoder = CouplingFeatureEncoder(
                output_dim=coupling_encoder_output_dim,
                embed_dim=coupling_encoder_embed_dim,
                num_heads=coupling_encoder_num_heads,
                dropout=dropout, # Reuse the global dropout
                pooling_method=coupling_encoder_pooling,
                chunk_size=coupling_encoder_chunk_size
            )
            coupling_encoder_final_output_dim = coupling_encoder_output_dim


        # PhiGnet-style GCN
        self.coupling_processor = None
        self.graph_conv = None
        phignet_output_dim = 0
        if self.use_phignet:
            self.coupling_processor = PhiGnetCouplingProcessor(
                cut_thresh=phignet_cut_thresh,
                use_binary_weights=phignet_binary_weights,
                normalize_before_threshold=phignet_normalization
            )
            self.graph_conv = PhiGnetGraphConv(
                input_dim=self.esm_output_dim,
                hidden_dim=phignet_gcn_hidden_dim,
                num_layers=phignet_gcn_layers,
                use_bias=False # PhiGnet's GraphConv layers had use_bias=False
            )
            phignet_output_dim = phignet_gcn_hidden_dim * phignet_gcn_layers


        # GraphEC Feature Extractor and Encoder
        self.feature_calculator = None
        self.graphec_encoder = None
        graphec_final_output_dim = 0
        if self.use_graphec:
            self.feature_calculator = GraphECFeatureCalculator()
            graphec_node_input_dim = self.esm_output_dim + dssp_dim + graphec_node_feature_dim
            self.graphec_encoder = GraphECEncoder(
                node_in_dim=graphec_node_input_dim,
                edge_in_dim=graphec_edge_feature_dim,
                hidden_dim=graphec_hidden_dim,
                output_dim=graphec_output_dim,
                num_layers=graphec_layers,
                drop_rate=graphec_dropout
            )
            graphec_final_output_dim = graphec_output_dim

        
        # Calculate the combined dimension of all protein-side features
        protein_feature_dim = self.esm_output_dim + structure_output_dim
        if self.use_coupling_encoder:
            protein_feature_dim += coupling_encoder_final_output_dim
        if self.use_phignet:
            protein_feature_dim += phignet_output_dim
        if self.use_dssp:
            protein_feature_dim += dssp_dim
        if self.use_graphec:
            protein_feature_dim += graphec_final_output_dim


        # reaction
        if self.rxn_attn_model:
            self.brige_model = Predictior(
                in_dim=protein_feature_dim, 
                out_dim=self.rxn_attn_model.node_out_dim
            )
            self.interaction_net = GlobalMultiHeadAttention(
                d_model=self.rxn_attn_model.node_out_dim,
                heads=cross_attention_heads,
                n_layers=cross_attention_layers,
                cross_attn_h_rate=cross_attention_h_rate,
                dropout=cross_attention_dropout,
                positional_number=positional_number,
            )
            self.prediction_head = Predictior(
                in_dim=self.rxn_attn_model.node_out_dim,
                out_dim=classification_output_dim
            )
        else:
            # If no reaction model, prediction is directly from fused protein features
            self.brige_model = None
            self.interaction_net = None
            self.prediction_head = Predictior(
                in_dim=protein_feature_dim,
                out_dim=classification_output_dim
            )

        self.to(device)



    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        graph = batch["protein_graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        protein_features_list = []

        # Sequence Features
        output_esm = self.esm_model(graph, graph.node_feature.float())
        node_feature_seq = output_esm.get("residue_feature")

        # Evolutionary Feature Modulation
        # ConditionNet modifies the sequence features in-place before they are used by other modules.
        if self.condition_net_version == "v2":
            packed_seq_feats, protein_mask = pack_residue_feats(graph, node_feature_seq)
            padded_conservation_scores, _ = pack_residue_feats(graph, batch["conservation_scores"])
            conditioned_embeddings_packed = self.condition_network(
                fused_embeddings=packed_seq_feats,
                conservation_scores=padded_conservation_scores,
                dca_single_potentials=batch["single_site_potentials"],
                coupling_matrix=batch["coupling_matrices"],
                valid_tokens_mask=protein_mask.bool(),
            )
            sequence_feature = conditioned_embeddings_packed[protein_mask.bool()]
        else:
            sequence_feature = node_feature_seq
        
        protein_features_list.append(sequence_feature)
        
        # Structure Features
        output_struct = self.structure_model(graph, node_feature_seq)
        node_feature_struct = output_struct.get("node_feature")
        protein_features_list.append(node_feature_struct)

        # Evolutionary Features
        if self.coupling_feature_encoder:
            node_feature_coupling = self.coupling_feature_encoder(
                dca_single_potentials=batch["single_site_potentials"],
                coupling_matrix=batch["coupling_matrices"]
            )
            protein_features_list.append(node_feature_coupling)

        # Evolutionary Features
        if self.use_phignet:
            edge_index, edge_weight, _ = self.coupling_processor(
                batch["coupling_matrices"],
                graph.num_residues
            )
            node_feature_phignet = self.graph_conv(
                node_feature_seq,
                edge_index,
                edge_weight
            )
            protein_features_list.append(node_feature_phignet)

        # DSSP Features
        if self.use_dssp:
            protein_features_list.append(batch["dssp_features"])

        # GraphEC Features
        if self.use_graphec:
            graphec_node_feature, graphec_edge_feature = self.feature_calculator(
                graph=graph,
                atom_coords=batch["atom_coords"],
                edge_index=batch["batched_graphec_edge_indices"]
            )
            graphec_h_v = torch.cat([
                node_feature_seq,
                batch["dssp_features"],
                graphec_node_feature
            ], dim=-1)
            graphec_output_feature = self.graphec_encoder(
                h_V=graphec_h_v,
                edge_index=batch["batched_graphec_edge_indices"],
                h_E=graphec_edge_feature,
                batch_id=batch["graphec_batch_id"]
            )
            protein_features_list.append(graphec_output_feature)

        # Concat
        fused_protein_feature = torch.cat(protein_features_list, dim=-1)

        # reaction
        if self.rxn_attn_model:
            output_reaction = self.rxn_attn_model(
                **batch["reaction_graph"],
                return_rts=True
            )
            substrate_node_feature, substrate_mask = output_reaction
            bridged_protein_feature = self.brige_model(fused_protein_feature)
            protein_feature_padded, protein_mask = pack_residue_feats(graph, bridged_protein_feature)
            
            # Cross-Attention
            final_feature_padded, _, _, _, _ = self.interaction_net(
                src=protein_feature_padded,
                tgt=substrate_node_feature,
                src_mask=protein_mask,
                tgt_mask=substrate_mask
            )
            final_feature = final_feature_padded[protein_mask.bool()]
        else:
            final_feature = fused_protein_feature
            _, protein_mask = pack_residue_feats(graph, final_feature)

        # Prediction
        logits = self.prediction_head(final_feature)
        
        return logits, protein_mask
    


    def _load_rxn_attn_model(self, model_state_path, from_scratch):
        if not model_state_path:
            return None
            
        model_state, rxn_attn_args = read_model_state(model_save_path=model_state_path)
        rxn_attn_model = ReactionMGMTurnNet(
            node_in_feats=rxn_attn_args["node_in_feats"],
            node_out_feats=rxn_attn_args["node_out_feats"],
            edge_in_feats=rxn_attn_args["edge_in_feats"],
            edge_hidden_feats=rxn_attn_args["edge_hidden_feats"],
            num_step_message_passing=rxn_attn_args["num_step_message_passing"],
            attention_heads=rxn_attn_args["attention_heads"],
            attention_layers=rxn_attn_args["attention_layers"],
            dropout=rxn_attn_args["dropout"],
            num_atom_types=rxn_attn_args["num_atom_types"],
            cross_attn_h_rate=rxn_attn_args["cross_attn_h_rate"],
            is_pretrain=False,
        )
        if not from_scratch:
            print(
                f"Loading reaction attention model checkpoint from {model_state_path}..."
            )
            rxn_attn_model = load_pretrain_model_state(rxn_attn_model, model_state)
        return rxn_attn_model
    


class RFdiffusionFeatureExtractor:

    def __init__(self, model: nn.Module, target_module_names: List[str]):
        """
        Args:
            model (nn.Module): The RoseTTAFoldModule instance.
            target_module_names (List[str]): A list of full module names to hook into.
                e.g., ['simulator.main_block.5', 'simulator.str_refiner']. To get the final output of the whole simulator, use ['simulator'].
        """
        self.features = {}
        self.current_t = None
        self.handles = []
        
        all_modules = dict(model.named_modules())
        
        for name in target_module_names:
            if name not in all_modules:
                raise ValueError(f"Module '{name}' not found in the model. Please check the name.")
            
            target_module = all_modules[name]
            
            handle = target_module.register_forward_hook(
                lambda module, inp, outp, module_name=name: self.save_hook(module, inp, outp, module_name)
            )
            self.handles.append(handle)

    def set_timestep(self, t: int):
        """
        Informs the extractor of the current diffusion timestep. This must be called
        before the forward pass for which features are to be captured.
        """
        self.current_t = t

    def save_hook(self, module: nn.Module, input: Tuple, output: Tuple, module_name: str):
        """
        {
            10: {
                'simulator.main_block.5': {'msa': ..., 'state': ...},
                'simulator.main_block.10': {'msa': ..., 'state': ...}
            },
            2: {
                'simulator.main_block.5': {'msa': ..., 'state': ...},
                'simulator.main_block.10': {'msa': ..., 'state': ...}
            }
        }
        The hook function. It saves features keyed by the current timestep and module name.
        This function now correctly handles different output formats from different modules.
        """
        if self.current_t is None:
            return

        if self.current_t not in self.features:
            self.features[self.current_t] = {}

        features_to_save = {'msa': None, 'pair': None, 'state': None}

        if isinstance(module, IterativeSimulator):
            # Output is (msa, pair, R_s, T_s, alpha_s, state), len=6
            if isinstance(output, tuple) and len(output) == 6:
                features_to_save['msa'] = output[0][:, 0].detach()
                features_to_save['pair'] = output[1].detach()
                features_to_save['state'] = output[5].detach()
            else:
                logging.warning(f"Unexpected output from IterativeSimulator '{module_name}'.")

        elif isinstance(module, IterBlock):
            # Output is (msa, pair, R, T, state, alpha), len=6
            if isinstance(output, tuple) and len(output) == 6:
                features_to_save['msa'] = output[0][:, 0].detach()
                features_to_save['pair'] = output[1].detach()
                features_to_save['state'] = output[4].detach()
            else:
                logging.warning(f"Unexpected output from IterBlock '{module_name}'.")

        elif isinstance(module, Str2Str):
            # Output is (Ri, Ti, state, alpha), len=4
            # This module does not output msa or pair features.
            if isinstance(output, tuple) and len(output) == 4:
                features_to_save['state'] = output[2].detach()
            else:
                logging.warning(f"Unexpected output from Str2Str module '{module_name}'.")
        
        else:
            logging.warning(f"Hooked on an unsupported module type: {type(module).__name__} for '{module_name}'.")

        self.features[self.current_t][module_name] = features_to_save


    def get_features(self) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        """Returns the nested dictionary of all captured features and clears the storage."""
        feats = self.features
        self.features = {}
        self.current_t = None # Reset timestep after getting features
        return feats

    def remove_hooks(self):
        """Removes all registered hooks to prevent memory leaks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __del__(self):
        """Ensure hooks are removed when the object is garbage collected."""
        self.remove_hooks()



class RFDSite(nn.Module):
    """
    掩码策略：diffusion_mask
        - 全0或者全1
        - 以一定百分概率随机掩码

    Index:
        - 最大化已知信息输入（问题描述：RoseTTAFoldModule模块只能处理主链原子，有什么办法可以让RoseTTAFoldModule看到侧链信息）
            - 将原始无噪结构作为模板
            - 真实MSA（尚未实现）
    
    checkpoint: https://aistudio.google.com/prompts/10uHMcDOYO8D1zvg46J-hLumsqUqm535d
    """
    def __init__(self, conf: DictConfig):
        super().__init__()
        self._conf = conf
        self._device = torch.device(self._conf.device)

        ################################
        ### Select Appropriate Model ###
        ################################
        if conf.inference.model_directory_path is not None:
            model_directory = conf.inference.model_directory_path
        else:
            model_directory = f"{SCRIPT_DIR}/../ckpt/pretrain"
        print(f"Reading models from {model_directory}")
        if conf.inference.ckpt_override_path is not None:
            self.ckpt_path = conf.inference.ckpt_override_path
            print("WARNING: You're overriding the checkpoint path from the defaults. Check th* at the model you're providing can run with the inputs you're providing.")
        else:
            # # 触发条件：
            # #     * inpaint_seq: 用户希望在生成结构的同时，让模型重新设计某段区域的氨基酸序列。
            # #     * provide_seq: 用户在进行“局部扩散”时，不仅提供了结构，还提供了序列信息，允许模型对序列也进行扰动。
            # #     * inpaint_str: 用户希望模型在一个大的蛋白质框架内重新生成（修复）一整段结构。
            # # 选择逻辑：只要满足以上任一条件，就意味着任务涉及“填空”，需要使用专门为此训练的 Inpainting（修复）模型。
            # #     * InpaintSeq_ckpt.pt：标准的修复模型。
            # #     * InpaintSeq_Fold_ckpt.pt：如果同时还启用了 scaffoldguided（支架引导）模式，会选择这个更特殊的模型。它不仅擅长修复，还能更好地理解和利用二级结构信息。
            # if conf.contigmap.inpaint_seq is not None or conf.contigmap.provide_seq is not None or conf.contigmap.inpaint_str:
            #     # use model trained for inpaint_seq
            #     if conf.contigmap.provide_seq is not None:
            #         # this is only used for partial diffusion
            #         assert conf.diffuser.partial_T is not None, "The provide_seq input is specifically for partial diffusion"
            #     if conf.scaffoldguided.scaffoldguided:
            #         self.ckpt_path = f'{model_directory}/InpaintSeq_Fold_ckpt.pt'
            #     else:
            #         self.ckpt_path = f'{model_directory}/InpaintSeq_ckpt.pt'
            # elif conf.ppi.hotspot_res is not None and conf.scaffoldguided.scaffoldguided is False:
            #     # use complex trained model
            #     self.ckpt_path = f'{model_directory}/Complex_base_ckpt.pt'
            # elif conf.scaffoldguided.scaffoldguided is True:
            #     # use complex and secondary structure-guided model
            #     self.ckpt_path = f'{model_directory}/Complex_Fold_base_ckpt.pt'
            # else:
            #     # use default model
            #     self.ckpt_path = f'{model_directory}/Base_ckpt.pt'
            self.ckpt_path = f'{model_directory}/Base_ckpt.pt'
        # for saving in trb file:
        # assert self._conf.inference.trb_save_ckpt_path is None, "trb_save_ckpt_path is not the place to specify an input model. Specify in inference.ckpt_override_path"
        self._conf['inference']['trb_save_ckpt_path']=self.ckpt_path


        #######################
        ### load model
        #######################
        self.load_checkpoint()
        self.assemble_config_from_chk()
        self.model = self.load_model()
        
        # 冻结 RFdiffusion 模型的参数，我们只将其用作特征提取器
        for param in self.model.parameters():
            param.requires_grad = False



        #################################
        ### Initialize helper objects ###
        #################################
        self.inf_conf = self._conf.inference
        self.denoiser_conf = self._conf.denoiser
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        self.predictor_type = self._conf.predictor_type
        if conf.inference.schedule_directory_path is not None:
            schedule_directory = conf.inference.schedule_directory_path
        else:
            schedule_directory = f"{SCRIPT_DIR}/../ckpt/schedules"
        if not os.path.exists(schedule_directory):
            os.mkdir(schedule_directory)
        self.diffuser = Diffuser(**self._conf.diffuser, cache_dir=schedule_directory, device=self._device)
        self.allatom = ComputeAllAtomCoords().to(self._device)



        
        # RFdiffusion 特征提取器
        self.feature_blocks = self._conf.feature_blocks
        self.feature_timesteps = self._conf.feature_timesteps
        self.feature_extractor = RFdiffusionFeatureExtractor(self.model, self.feature_blocks)
        
        # Sequence Feature Extractor
        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model="ESM-2-650M")
        self.esm_output_dim = self.esm_model.output_dim
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # Graph Construction
        if self._conf.use_graph_construction:
            self.graph_construction_model = layers.geometry.GraphConstruction(
                node_layers=[layers.geometry.AlphaCarbonNode()],
                edge_layers=[
                    layers.geometry.SequentialEdge(max_distance=2),
                    layers.geometry.SpatialEdge(radius=10, min_distance=5),
                    layers.geometry.KNNEdge(k=10, max_distance=0),
                ],
                edge_feature="gearnet"
            )
        else:
            self.graph_construction_model = None

        if self._conf.use_gearnet:
            # Structure Feature Extractor
            self.structure_model = StructureEncoder(
                input_dim=self.esm_output_dim,
                hidden_dims=[512, 512, 512, 512, 512, 512],
                num_relation=7,
                edge_input_dim=None,
                num_angle_bin=None,
                batch_norm=True,
                short_cut=True,
                readout="sum",
            )
            structure_output_dim = 512
        else:
            self.structure_model = None
            structure_output_dim = 0

        # Reaction Feature Extractor
        self.rxn_attn_model = self._load_rxn_attn_model(self._conf.rxn_model_path, False)


        # 计算维度
        d_state = self._conf.model.SE3_param_full['l0_out_features']
        d_msa = self._conf.model.d_msa
        d_pair = self._conf.model.d_pair
        rfd_feature_dim = len(self.feature_timesteps) * len(self.feature_blocks) * (d_msa + d_state)
        rfd_pair_feature_dim = len(self.feature_timesteps) * len(self.feature_blocks) 
        protein_feature_dim = rfd_feature_dim + structure_output_dim + self.esm_output_dim

        # Fusion
        self.brige_model = Predictior(
            in_dim=protein_feature_dim,
            out_dim=self.rxn_attn_model.node_out_dim
        )
        self.interaction_net = GlobalMultiHeadAttention(
            d_model=self.rxn_attn_model.node_out_dim,
            heads=8,
            n_layers=3,
            cross_attn_h_rate=1.0,
            dropout=0.1 if self.training else 0.0,
            positional_number=0,
        )
        if self.predictor_type == "Predictior":
            self.prediction_head = Predictior(
                in_dim=self.rxn_attn_model.node_out_dim,
                out_dim=4
            )
        elif self.predictor_type == "PairBiasedAttentionPredictor":
            self.prediction_head = PairBiasedAttentionPredictor(
                node_dim=self.rxn_attn_model.node_out_dim,
                pair_dim=rfd_pair_feature_dim,
                hidden_dim=self.rxn_attn_model.node_out_dim * 2,
                output_dim=4, # 分类数
                num_layers=4,
                num_heads=4,
                dropout=0.1 if self.training else 0.0
            )




    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        x_t, seq_t, diffusion_mask, valid_residue_mask, idx_rf, xyz_27 = self.sample_init_batch(batch)
        for t in range(self.t_step_input, self.inf_conf.final_step - 1, -1):
            if t in self.feature_timesteps:
                self.feature_extractor.set_timestep(t) # 通知提取器当前时间步

            _, x_t, _, _ = self.sample_step_batch(
                t=t,
                x_t=x_t,
                seq_t=seq_t, 
                diffusion_mask=diffusion_mask,
                valid_residue_mask=valid_residue_mask,
                idx_rf=idx_rf,
                xyz_27=xyz_27
            )
        all_captured_features = self.feature_extractor.get_features()
        
        per_residue_features_list = []
        per_pair_features_list = []
        for t in sorted(self.feature_timesteps, reverse=True):
            for block_name in self.feature_blocks:
                features = all_captured_features.get(t, {}).get(block_name, {})
                # 从 hook 捕获的特征已经是 (B, L, D) 形状，可以直接使用
                msa_feat = features.get('msa')
                state_feat = features.get('state')
                pair_feat = features.get('pair')
                if msa_feat is not None and state_feat is not None:
                    # 将 msa 和 state 特征拼接
                    fused = torch.cat([msa_feat, state_feat], dim=-1)
                    per_residue_features_list.append(fused)
                    per_pair_features_list.append(pair_feat)
                else:
                    raise RuntimeError(f"Features for t={t}, block={block_name} not found.")
        rfd_features = torch.cat(per_residue_features_list, dim=-1)
        rfd_pair_features = torch.cat(per_pair_features_list, dim=-1) # 拼接 pair 特征
        protein_features_list = [rfd_features]

        # ESM feature
        graph = batch["protein_graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        esm_output = self.esm_model(graph, graph.node_feature.float())
        esm_features_unpadded = esm_output['residue_feature']
        esm_features_padded, protein_mask = pack_residue_feats(graph, esm_features_unpadded)
        assert protein_mask.shape == valid_residue_mask.shape, f"Protein mask shape {protein_mask.shape} does not match valid_residue_mask shape {valid_residue_mask.shape}"
        assert esm_features_padded.shape[1] == valid_residue_mask.shape[1], f"ESM features length {esm_features_padded.shape[1]} does not match L_max {valid_residue_mask.shape[1]}"
        protein_features_list.append(esm_features_padded)

        # GearNet
        if self.structure_model:
            structure_output = self.structure_model(graph, esm_features_unpadded)
            struct_features_unpadded = structure_output['node_feature']
            struct_features_padded, _ = pack_residue_feats(graph, struct_features_unpadded)
            assert struct_features_padded.shape[1] == valid_residue_mask.shape[1], f"Structure features length {struct_features_padded.shape[1]} does not match L_max {valid_residue_mask.shape[1]}"
            protein_features_list.append(struct_features_padded)

        # 拼接所有蛋白质侧特征
        protein_total_features = torch.cat(protein_features_list, dim=-1)

        # 交叉注意力
        bridged_protein_feature = self.brige_model(protein_total_features)
        if self.rxn_attn_model:
            substrate_node_feature, substrate_mask = self.rxn_attn_model(**batch['reaction_graph'], return_rts=True)
            final_feature_padded, _, _, _, _ = self.interaction_net(
                src=bridged_protein_feature,
                tgt=substrate_node_feature,
                src_mask=valid_residue_mask,
                tgt_mask=substrate_mask
            )
        else:
            raise ValueError("DiffSite requires a reaction model to be specified via rxn_model_path.")


        # (|V_res|, D)
        if self.predictor_type == "Predictior":
            final_features_unpadded = final_feature_padded[valid_residue_mask]
            logits = self.prediction_head(final_features_unpadded)
        elif self.predictor_type == "PairBiasedAttentionPredictor":
            logits_padded = self.prediction_head(
                node_features=final_feature_padded,
                pair_features=rfd_pair_features,
                mask=valid_residue_mask
            )
            logits = logits_padded[valid_residue_mask]

        outputs = {
            "logits": logits,
            "valid_mask": valid_residue_mask,
            "final_features": final_feature_padded,
            "substrate_features": substrate_node_feature,
            "substrate_mask": substrate_mask,
        }
        return outputs




    def sample_init_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched analogue of ``Sampler.sample_init`` specialised for the active-site setting, where contigs collapse to a single continuous chain (例如 ``['A1-300/0']``) and no sequence / structure inpainting is requested.
        
        $$ seq_orig的氨基酸数字索引映射和ESM的氨基酸数字索引映射可能不一致

        Args:
            batch (Dict[str, Any]): A dictionary containing the batched input data, typically from a dataloader. Must contain an 'x_0' key with packed protein features.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing the padded tensors ready for the first step of reverse diffusion.
                - xt_padded (torch.Tensor): Padded noisy coordinates for each protein. Shape: `(B, L_max, 14, 3)`.
                - seq_t_padded (torch.Tensor): Padded one-hot encoded sequences. Shape: `(B, L_max, 22)`.
                - diffusion_mask_padded (torch.Tensor): The unified integer state mask. Shape: `(B, L_max)`.
                    - `0`: Diffusible residue.
                    - `1`: Fixed (motif) residue.
                    - `2`: Padding residue.
                - valid_residue_mask (torch.Tensor): A boolean mask indicating valid (real) residues. Shape: `(B, L_max)`.
                    - `True`: Real residue.
                    - `False`: Padding residue.
                - idx_rf_padded (torch.Tensor): Padded residue indices, with padding marked by -1. Shape: `(B, L_max)`.
        """

        assert "x_0" in batch, "Batch must contain an 'x_0' entry produced by the collator."
        device = self._device

        # =============================
        # 读取批次化特征：所有残基已拼接成一个“大样本”
        # =============================
        x0 = batch["x_0"]
        xyz_27 = x0["xyz_27"].to(device=device, dtype=torch.float32)                 # (N, 27, 3), N = total_residues in batch
        mask_27 = x0["mask_27"].to(device=device, dtype=torch.bool)                  # (N, 27)
        seq_orig = x0["seq"].to(device=device, dtype=torch.long)                     # (N)
        lengths = x0["seq_shapes"].to(device=device, dtype=torch.long).view(-1)      # (B,)

        assert lengths.numel() > 0, "Empty batch passed to sample_init_batch."
        assert torch.equal(lengths, batch["protein_graph"].num_residues.to(device=device, dtype=torch.long)), ("lengths and batch['protein_graph'].num_residues are inconsistent, which may lead to feature segmentation errors.")
        total_residues = int(lengths.sum())
        assert total_residues == seq_orig.numel(), "Packed residue count mismatch."


        # =============================
        # 2. 动态生成 diffusion_mask
        # =============================
        mask_conf = self._conf.diffusion_mask
        
        if mask_conf.strategy == 'all_diffuse':
            # 策略一：全部扩散 (全0)
            # 含义: 将整个蛋白质视为一个需要整体优化的动态实体。
            diffusion_mask = torch.zeros(total_residues, dtype=torch.bool, device=device)
        
        elif mask_conf.strategy == 'all_fixed':
            # 策略二：全部固定 (全1)
            # 含义: 完全不进行结构扩散，只在t=T时加一次噪声，然后在去噪时将整个结构作为固定的motif。
            diffusion_mask = torch.ones(total_residues, dtype=torch.bool, device=device)
            
        elif mask_conf.strategy == 'random':
            # 策略三：按概率随机掩码（蛋白质独立）
            prob = mask_conf.get('prob', 0.15) # 获取固定的概率
            masks_list = [
                torch.rand(L, device=device) < prob 
                for L in lengths
            ]
            diffusion_mask = torch.cat(masks_list)
        else:
            raise ValueError(f"Unknown diffusion_mask strategy: {mask_conf.strategy}")
        

        # =============================
        # 序列初始化：partial_T 保留真实序列，否则填充掩码 token
        # =============================
        if self.diffuser_conf.partial_T:
            seq_tokens = seq_orig.clone()                                             # (N)
        else:
            seq_tokens = torch.full_like(seq_orig, 21)
        seq_t = F.one_hot(seq_tokens, num_classes=22).to(torch.float32)               # (N, 22)

        # =============================
        # 坐标初始化：支持部分扩散与从模板起点的全扩散
        # =============================
        if self.diffuser_conf.partial_T:
            xyz_for_diffusion = xyz_27.clone()  # $$这里xyz_27[:, 14:, :]是nan，可能有问题
        else:
            nan_template = torch.full((1,1,total_residues,27,3), float('nan'), device=device, dtype=xyz_27.dtype)
            xyz_for_diffusion = get_init_xyz(nan_template).squeeze(0).squeeze(0)
        atom_mask_mapped = mask_27

        # =============================
        # 前向扩散：一次性对大样本施加噪声
        # =============================
        self.t_step_input = int(self.diffuser_conf.partial_T) if self.diffuser_conf.partial_T else int(self.diffuser_conf.T)

        # 扩散
        fa_stack, _ = self.diffuser.batch_diffuse_pose(
            xyz_for_diffusion,
            seq_t,
            atom_mask_mapped,
            diffusion_mask,
            lengths,
            t_final=self.t_step_input,
            include_motif_sidechains=True,
        )
        xt = fa_stack[:, :14, :].contiguous()  # (N,14,3)

        # =============================
        # 状态缓存
        # =============================
        self.lengths = lengths
        self.potential_manager = None
        self.T = int(self.diffuser_conf.T)

        # 初始化 denoiser
        denoise_kwargs = OmegaConf.to_container(self.diffuser_conf)
        denoise_kwargs.update(OmegaConf.to_container(self.denoiser_conf))
        denoise_kwargs.update({
            'L': total_residues,
            'diffuser': self.diffuser,
            'potential_manager': self.potential_manager,
            'device': device
        })
        self.denoiser = iu.Denoise(**denoise_kwargs)


        # ====================================================================================
        # 将 "拼接式" (Packed) 批量数据转换为 "填充式" (Padded) 批量数据
        # ====================================================================================
        processed_idx = x0['processed_idx'].to(device=device, dtype=torch.long)
        

        idx_unpadded_list = list(processed_idx.split(self.lengths.tolist()))
        xt_list = list(torch.split(xt, self.lengths.tolist()))
        seq_t_list = list(torch.split(seq_t, self.lengths.tolist()))
        diffusion_mask_list = list(torch.split(diffusion_mask.long(), self.lengths.tolist()))
        xyz_27_list = list(torch.split(xyz_27, self.lengths.tolist()))
        
        xt_padded = torch.nn.utils.rnn.pad_sequence(xt_list, batch_first=True, padding_value=0.0)
        seq_t_padded = torch.nn.utils.rnn.pad_sequence(seq_t_list, batch_first=True, padding_value=0.0)
        diffusion_mask_padded = torch.nn.utils.rnn.pad_sequence(diffusion_mask_list, batch_first=True, padding_value=2).long()  # 创建并填充 `diffusion_mask`，并指定填充值为 2. 这将创建一个整数张量，其中真实残基的值为 0 (可扩散) 或 1 (固定motif)，而填充残基的值为 2。
        idx_padded = torch.nn.utils.rnn.pad_sequence(idx_unpadded_list, batch_first=True, padding_value=-1)
        xyz_27_padded = torch.nn.utils.rnn.pad_sequence(xyz_27_list, batch_first=True, padding_value=float('nan')) # 使用 NaN 填充


        valid_residue_mask = (diffusion_mask_padded != 2) # True for real residues, False for padding

        self.mask_str = diffusion_mask_padded   # 更新 self.mask_str 以便在 batch_preprocess 中使用。这里我们传递 diffusion_mask_padded 本身，因为它现在包含了所有状态信息。batch_preprocess 内部将需要根据 0, 1, 2 的值来派生出不同的行为。

        return xt_padded, seq_t_padded, diffusion_mask_padded, valid_residue_mask, idx_padded, xyz_27_padded



    def sample_step_batch(
        self, t: int,
        x_t: torch.Tensor,
        seq_t: torch.Tensor, 
        diffusion_mask: torch.Tensor,
        valid_residue_mask: torch.Tensor,
        idx_rf: torch.Tensor,
        xyz_27: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行单步反向扩散（去噪）的批量化版本，替代 Sampler.sample_step。
        
        此函数接收一个批次的噪声结构和序列，通过调用底层的 RoseTTAFold 模型预测出去噪后的结构 (p_x0)，
        然后利用扩散公式计算出下一个时间步 t-1 的噪声结构 x_{t-1}。

        Args:
            t (int): 当前时间步。
            x_t (torch.Tensor): 当前批次的噪声结构坐标，形状为 `(B, L_max, 14, 3)`。
            seq_t (torch.Tensor): 当前批次的序列 one-hot 编码，形状为 `(B, L_max, 22)`。
            diffusion_mask (torch.Tensor): 统一的整数状态掩码，形状为 `(B, L_max)`。
                - `0`: 表示该残基是可扩散的 (Diffusible)。
                - `1`: 表示该残基是固定的motif (Fixed/Motif)。
                - `2`: 表示该位置是填充的 (Padding)。
            valid_residue_mask (torch.Tensor): 布尔掩码，标记批次中的有效（真实）残基，形状为 `(B, L_max)`。
                - `True`: 真实残基。
                - `False`: 填充残基。
            idx_rf (torch.Tensor): 批次化的残基索引，形状为 `(B, L_max)`。
            xyz_27 (torch.Tensor): 批次化的真实全原子坐标，形状为 `(B, L_max, 27, 3)`。

        Returns:
            Tuple: 包含
                - px0 (torch.Tensor): 预测的无噪结构 (p_x0)，形状为 `(B, L_max, 14, 3)`。
                - x_t_1 (torch.Tensor): 为下一个时间步准备的噪声结构 (x_{t-1})，形状为 `(B, L_max, 14, 3)`。
                - seq_t_1 (torch.Tensor): 下一步的序列（通常与输入 seq_t 相同），形状为 `(B, L_max, 22)`。
                - plddt (torch.Tensor): 每个残基预测的 pLDDT 值，形状为 `(B, L_max)`。
        """
        ##################################
        ######## 准备模型输入 ###########
        ##################################

        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t_template, alpha_t = self.batch_preprocess(seq_t, x_t, t, idx_rf, diffusion_mask, valid_residue_mask, xyz_27)
        B,N,L = xyz_t_template.shape[:3]


        ##################################
        ######## Str Self Cond ###########
        ##################################
        if (t < self.diffuser.T) and (t != self.diffuser_conf.partial_T):   
            zeros = torch.zeros(B, 1, L, 24, 3, dtype=torch.float32, device=x_t.device)
            xyz_t_self_cond = torch.cat((self.prev_pred.unsqueeze(1), zeros), dim=-2) # [B,1,L,27,3]
            t2d_44 = xyz_to_t2d(xyz_t_self_cond) # [B,1,L,L,44]
        else:
            t2d_44 = torch.zeros_like(t2d[...,:44])
        # No effect if t2d is only dim 44
        t2d[...,:44] = t2d_44


        ####################
        ### Forward Pass ###
        ####################
        print("DEBUG-1")
        with torch.no_grad():
            msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
                                msa_masked,
                                msa_full,
                                seq_in,
                                xt_in,
                                idx_pdb,
                                t1d=t1d,
                                t2d=t2d,
                                xyz_t=xyz_t_template,
                                alpha_t=alpha_t,
                                msa_prev = None,
                                pair_prev = None,
                                state_prev = None,
                                t=torch.tensor(t, device=x_t.device),
                                return_infer=True,
                                motif_mask=(diffusion_mask != 0), # 模型需要固定的部分是 MOTIF (值为1) 和 PADDING (值为2)
                                attention_mask=valid_residue_mask)            
        self.prev_pred = torch.clone(px0)
        print("DEBUG-2")

        ####################
        ### prediction of X0
        ####################
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0    = px0.squeeze(1)[:, :, :14] # (B, L, 14, 3)


        ####################
        ### 计算 x_{t-1} ###
        ####################
        if t > self.inf_conf.final_step:
            xt_flat = x_t[valid_residue_mask]
            px0_flat = px0[valid_residue_mask]
            motif_mask_flat = (diffusion_mask == 1)[valid_residue_mask]
            lengths = self.lengths.to(x_t.device)

            x_t_1_flat, px0_aligned_flat = self.denoiser.batch_get_next_pose(
                xt=xt_flat,
                px0=px0_flat,
                t=t,
                diffusion_mask=motif_mask_flat,
                lengths=lengths,
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input
            )

            x_t_1 = x_t.clone()
            x_t_1[valid_residue_mask] = x_t_1_flat

            px0_padded = px0.clone()
            px0_padded[valid_residue_mask] = px0_aligned_flat
            px0 = px0_padded
            
            seq_t_1 = seq_t
        else:
            x_t_1 = px0.clone()
            seq_t_1 = seq_t.clone()

        return px0, x_t_1, seq_t_1, plddt




    def batch_preprocess(self, seq, xyz_t, t, idx_rf, diffusion_mask, valid_residue_mask, xyz_27, repack=False):
        
        """
        _preprocess函数现已支持GPU批量处理
        Function to prepare inputs to diffusion model. 将当前扩散步骤的“原始”数据（序列、噪声结构坐标、时间步）转换成 RoseTTAFoldModule 神经网络能够理解和处理的、高维度的特征张量（Feature Tensors）
        Args:
            seq: (B,L,22) one-hot sequence. One-hot 编码的序列。L 是蛋白质的总长度。22个类别代表：20种标准氨基酸 + 1个未知/Mask标记 + 1个特殊标记（通常用于标记不进行扩散的区域，如motif）。
            xyz_t: (B,L,14,3). 带噪声的结构坐标。这是在时间步 t 的结构，其中每个残基包含14个原子（N, CA, C, O, CB等骨架和β碳原子）的x,y,z坐标。
            t: current timestep. 当前时间步t。
            idx_rf (torch.Tensor): (B, L) 残基索引批次。
            diffusion_mask (torch.Tensor): (B, L) 统一的整数状态掩码。
                - `0`: 可扩散残基。
                - `1`: 固定motif残基。
                - `2`: 填充残基。
            valid_residue_mask (torch.Tensor): (B, L) 布尔掩码，标记有效（真实）残基。
                - `True`: 真实残基。
                - `False`: 填充残基。
            xyz_27: (B,L,27,3). 真实的全原子坐标。包含每个残基的27个原子（包括侧链）的x,y,z坐标。对于序列填充位置，坐标值为NaN。对于事实上不存在的原子的位置，坐标值见数据集类
            repack: if True, will only return features needed for repacking. 如果为True，则仅返回重新打包所需的特征。
        Returns:
            msa_masked: (B,1,L,48). 模拟的MSA（多序列比对）特征。由于推理时通常没有真实的MSA，代码通过复制序列信息来构建一个“伪MSA”，作为模型的输入。
            msa_full (B,1,L,25). 另一个模拟的MSA特征，用于模型的另一部分（full_emb）。
            seq: (B,L,22). 增加了批次维度的原始输入序列。
            xt_in: (B, L, 27, 3). 处理后的坐标，用于模型内部的几何计算。
            idx: (B,L). 残基索引映射。告诉模型每个残基在原始PDB、链等中的位置信息。
            t1d:(B, 1, L, 22/24)	torch.Tensor	一维特征。包含了序列信息和关键的时间步编码。维度会根据配置（如是否加入hotspot）而变化。
            t2d:(B, 1, L, L, 44)	torch.Tensor	二维特征（配对特征）。从 xyz_t 计算得来，描述了所有残基对之间的几何关系（距离、方向等）。
            xyz_t_template:(B, 1, L, 27, 3)	torch.Tensor	模板坐标。这是经过处理（添加NaN、扩展维度）的 xyz_t，将作为模板输入给模型。
            alpha_t:(B, 1, L, 30)	torch.Tensor	扭转角特征。从 xyz_t 计算出的骨架和侧链扭转角（torsion angles），也是一种重要的几何输入。
        """
        B, L = seq.shape[:2]
        device = seq.device

        ##################
        ### msa_masked ###
        ##################
        # $$这里的MSA可以换成真实的MSA
        msa_masked = torch.zeros((B, 1, L, 48), device=device)
        seq_unsqueezed = seq.unsqueeze(1)
        msa_masked[:, :, :, :22] = seq_unsqueezed
        msa_masked[:, :, :, 22:44] = seq_unsqueezed
        msa_masked[:,:,0,46] = 1.0
        msa_masked[:,:,-1,47] = 1.0

        ################
        ### msa_full ###
        ################
        msa_full = torch.zeros((B, 1, L, 25), device=device)
        msa_full[:, :, :, :22] = seq_unsqueezed
        msa_full[:,:,0,23] = 1.0
        msa_full[:,:,-1,24] = 1.0

        ###########
        ### t1d ###
        ###########
        t1d = seq[..., :21].clone()
        mask_is_masked_token = (seq[..., 21] == 1) # Shape: (B, L)
        t1d[..., 20][mask_is_masked_token] = 1.0     # Shape: (B, L, 21)
        
        # Set timestep feature to 1 where diffusion mask is True, else 1-t/T
        # 为每个残基创建一个特征：
        #   * 对于需要扩散/生成的区域 (~self.mask_str)，特征值为 1 - t/T。这个值会随着 t 从 T 降到 1 而从 0 变化到接近 1。这等于告诉模型当前的“噪声水平”或“去噪进度”。
        #   * 对于固定不变的区域（如motif，self.mask_str），特征值恒为 1。这等于告诉模型：“这部分是基准，没有噪声，请参考它”。
        timefeature = torch.zeros((B, L), device=device)
        timefeature[diffusion_mask == 1] = 1.0  # MOTIF
        timefeature[diffusion_mask == 0] = 1.0 - t / self.T # DIFFUSIBLE
        # PADDING 状态 (值为2) 的 timefeature 保持为 0.0

        t1d = torch.cat((t1d, timefeature.unsqueeze(-1)), dim=-1).float()
        t1d = t1d.unsqueeze(1) # Shape: (B, L, 22) -> (B, 1, L, 22)

        #############
        ### xyz_t ###
        #############
        # xyz_t_processed = xyz_t.clone()
        # # 这里有选择地将某些残基的特定原子坐标设置为 NaN（Not a Number）来控制模型能“看到”哪些结构信息。
        # if self.preprocess_conf.sidechain_input:
        #     pass
        #     # mask_unknown = (diffusion_mask == 0) & (seq[..., 21] == 1)
        #     # xyz_t_processed[mask_unknown, 3:, :] = float('nan')
        # # 默认情况下 (sidechain_input: False)，所有正在被扩散的残基的侧链原子坐标都会被设为 NaN。这意味着在去噪过程中，模型主要依赖骨架信息，而侧链则由模型根据化学规则和学习到的知识自行预测和构建。
        # # self.mask_str: 这是一个布尔张量（mask），标记了哪些残基是固定不变的（例如，输入的蛋白质骨架或motif），这些区域的值为 True。
        # # ~self.mask_str.squeeze(): ~ 操作符将布尔值反转。因此，这个表达式选中了所有需要被扩散和生成的残基。
        # # xyz_t[..., 3:, :]: xyz_t 此时的形状是 (L, 14, 3)。第二个维度14代表14个原子。原子顺序通常是 N, CA, C, O, CB, ... (氮，α-碳，羰基碳，氧，β-碳等)。索引 3: 表示从第4个原子（也就是氧原子 'O'）开始，到最后一个原子的所有原子。这包括了部分骨架（氧）和整个侧链（从CB开始）。
        # else:
        #     xyz_t_processed[:, ~self.mask_str.squeeze(), 3:, :] = float('nan')

        # xyz_t_processed[diffusion_mask == 2, :, :] = float('nan')
        
        # xyz_t_template = xyz_t_processed.unsqueeze(1) # Shape: (B, 1, L, 14, 3)     这里可以修改_preprocess函数的输入：传入的xyz_t形状为 (B,L,27,3)，而不是 (B,L,14,3)，并且将(B,L,27,3)作为xyz_t_template进行下面的操作。但是有一个问题：xyz_to_t2d函数只处理了每个残基的C-N-CA原子，其他原子都没用上。如果要用到xyz_t的所有27个原子，就需要改写xyz_to_t2d函数。
        # xyz_t_template = torch.cat((xyz_t_template, torch.full((B, 1, L, 13, 3), float('nan'), device=device)), dim=3) # xyz_t_template Final Shape: (B, 1, L, 27, 3)，这是为了与 RoseTTAFold/AlphaFold2 的标准原子表示（最多27个原子）对齐。代码将 xyz_t 的14个原子和另外13个填充了 NaN 的原子拼接在一起，形成一个统一的 (..., 27, 3) 形状的原子坐标张量。这保证了无论氨基酸大小，输入到模型底层的原子数组形状都是固定的。
        xt_in = torch.cat((xyz_t, torch.full((B, L, 13, 3), float('nan'), device=device)),dim=2)    # (B,L,27,3)
        xyz_t_template = xyz_27.unsqueeze(1) # Shape: (B, 1, L, 27, 3)


        ###########
        ### t2d ###
        ###########
        # t2d: (B, T, L, L, 44)
        t2d = xyz_to_t2d(xyz_t_template)
        
        ###########      
        ### idx ###
        ###########
        # 直接使用从外部传入的、已计算好的 idx_rf
        idx = idx_rf

        ###############
        ### alpha_t ###
        ###############
        seq_tmp = t1d.squeeze(1)[..., :-1].argmax(dim=-1)
        tor_indices_dev = TOR_INDICES.to(device)
        tor_can_flip_dev = TOR_CAN_FLIP.to(device)
        ref_angles_dev = REF_ANGLES.to(device)
        alpha, _, alpha_mask, _ = util.get_torsions(xyz_t_template.reshape(B, L, 27, 3), seq_tmp, tor_indices_dev, tor_can_flip_dev, ref_angles_dev)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(B,1,L,10,2)
        alpha_mask = alpha_mask.reshape(B,1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, 1, L, 30)


        
        ######################
        ### added_features ###
        ######################
        if self.preprocess_conf.d_t1d >= 24:
            hotspot_tens = torch.zeros(B, L, device=device)
            if self.ppi_conf.hotspot_res is not None:
                hotspots = [(i[0], int(i[1:])) for i in self.ppi_conf.hotspot_res]
                hotspot_idx = []
                for i, res in enumerate(self.contig_map.con_ref_pdb_idx):
                    if res in hotspots:
                        hotspot_idx.append(self.contig_map.hal_idx0[i])
                
                if hotspot_idx:
                    hotspot_tens[:, hotspot_idx] = 1.0
            blank_feature = torch.zeros((B, 1, L, 1), device=device)
            hotspot_feature = hotspot_tens.unsqueeze(1).unsqueeze(-1) # (B,L) -> (B,1,L,1)
            t1d = torch.cat((t1d, blank_feature, hotspot_feature), dim=-1)

        return msa_masked, msa_full, seq, xt_in, idx, t1d, t2d, xyz_t_template, alpha_t
    


    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        print(f'Reading checkpoint from {self.ckpt_path}')
        print('This is inf_conf.ckpt_path')
        print(self.ckpt_path)
        self.ckpt  = torch.load(
            self.ckpt_path, map_location=self._device)

    def assemble_config_from_chk(self) -> None:
        """
        Function for loading model config from checkpoint directly.

        Takes:
            - config file

        Actions:
            - Replaces all -model and -diffuser items
            - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
        
        This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
        This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.

        """
        # # get overrides to re-apply after building the config from the checkpoint
        # overrides = []
        # if HydraConfig.initialized():
        #     overrides = HydraConfig.get().overrides.task
        print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

        for cat in ['model','diffuser','preprocess']:
            for key in self._conf[cat]:
                try:
                    print(f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {self.ckpt['config_dict'][cat][key]}")
                    self._conf[cat][key] = self.ckpt['config_dict'][cat][key]
                except:
                    pass
        
        # # add overrides back in again
        # for override in overrides:
        #     if override.split(".")[0] in ['model','diffuser','preprocess']:
        #         print(f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?') 
        #         mytype = type(self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
        #         self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])

    def load_model(self):
        """Create RosettaFold model from preloaded checkpoint."""
        
        # Read input dimensions from checkpoint.
        self.d_t1d=self._conf.preprocess.d_t1d
        self.d_t2d=self._conf.preprocess.d_t2d
        model = RoseTTAFoldModule(**self._conf.model, d_t1d=self.d_t1d, d_t2d=self.d_t2d, T=self._conf.diffuser.T).to(self._device)
        if self._conf.logging.inputs:
            pickle_dir = pickle_function_call(model, 'forward', 'inference')
            print(f'pickle_dir: {pickle_dir}')
        model = model.eval()
        print(f'Loading checkpoint.')
        model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        return model



    def _load_rxn_attn_model(self, model_state_path, from_scratch):
        if not model_state_path:
            return None
            
        model_state, rxn_attn_args = read_model_state(model_save_path=model_state_path)
        rxn_attn_model = ReactionMGMTurnNet(
            node_in_feats=rxn_attn_args["node_in_feats"],
            node_out_feats=rxn_attn_args["node_out_feats"],
            edge_in_feats=rxn_attn_args["edge_in_feats"],
            edge_hidden_feats=rxn_attn_args["edge_hidden_feats"],
            num_step_message_passing=rxn_attn_args["num_step_message_passing"],
            attention_heads=rxn_attn_args["attention_heads"],
            attention_layers=rxn_attn_args["attention_layers"],
            dropout=rxn_attn_args["dropout"],
            num_atom_types=rxn_attn_args["num_atom_types"],
            cross_attn_h_rate=rxn_attn_args["cross_attn_h_rate"],
            is_pretrain=False,
        )
        if not from_scratch:
            print(
                f"Loading reaction attention model checkpoint from {model_state_path}..."
            )
            rxn_attn_model = load_pretrain_model_state(rxn_attn_model, model_state)
        return rxn_attn_model



class MSASite(nn.Module):
    def __init__(
        self,
        esm_model_name: str = "ESM-2-650M",
        msa_model_name: str = "esm_msa1b_t12_100M_UR50S",
        
        # Structure model configuration
        structure_hidden_dims: List[int] = [512, 512, 512, 512, 512, 512],  # structure_hidden_dims[-1] is output dimension
        num_relation: int = 7,
        edge_input_dim: Optional[int] = None,
        num_angle_bin: Optional[int] = None,
        structure_batch_norm: bool = True,
        short_cut: bool = True,
        readout: str = "sum",

        # Reaction model configuration
        rxn_model_path: Optional[str] = None, # If None, the reaction branch is disabled
        from_scratch: bool = False,

        # Enzyme-Reaction Cross-attention configuration
        cross_attention_heads: int = 8,
        cross_attention_layers: int = 3,
        positional_number: int = 0,   # Not used, set to 0 to disable relative position feature
        cross_attention_h_rate: float = 1,     
        cross_attention_dropout: float = 0.1,
        
        # 输出层配置
        classification_output_dim: int = 4,
        
        # 通用参数
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_graph_construction: bool = True
    ):
        super(MSASite, self).__init__()
        self._device = device
        self.use_graph_construction = use_graph_construction

        # 图构建层
        if self.use_graph_construction:
            self.graph_construction_model = layers.geometry.GraphConstruction(
                node_layers=[layers.geometry.AlphaCarbonNode()],
                edge_layers=[
                    layers.geometry.SequentialEdge(max_distance=2),
                    layers.geometry.SpatialEdge(radius=10, min_distance=5),
                    layers.geometry.KNNEdge(k=10, max_distance=0),
                ],
                edge_feature="gearnet"
            )
        else:
            self.graph_construction_model = None

        # 序列特征提取器 (ESM)
        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=esm_model_name)
        self.esm_output_dim = self.esm_model.output_dim
        
        # 进化特征提取器 (MSA Transformer)
        self.msa_model = EvoMSATransformer(model_path=msa_model_name, device=device)
        self.msa_output_dim = self.msa_model.msa_out_dim


        # Structure Feature Extractor
        self.structure_model = StructureEncoder(
            input_dim=self.esm_output_dim + self.msa_output_dim,
            hidden_dims=structure_hidden_dims,
            num_relation=num_relation,
            edge_input_dim=edge_input_dim,
            num_angle_bin=num_angle_bin,
            batch_norm=structure_batch_norm,
            short_cut=short_cut,
            readout=readout,
        )
        structure_output_dim = structure_hidden_dims[-1]
        

        # Reaction Feature Extractor
        if rxn_model_path:
            self.rxn_attn_model = self._load_rxn_attn_model(rxn_model_path, from_scratch)
        else:
            self.rxn_attn_model = None

        # reaction
        if self.rxn_attn_model:
            self.brige_model = Predictior(
                in_dim=self.esm_model.output_dim + self.msa_model.msa_out_dim + structure_output_dim, 
                out_dim=self.rxn_attn_model.node_out_dim
            )
            self.interaction_net = GlobalMultiHeadAttention(
                d_model=self.rxn_attn_model.node_out_dim,
                heads=cross_attention_heads,
                n_layers=cross_attention_layers,
                cross_attn_h_rate=cross_attention_h_rate,
                dropout=cross_attention_dropout,
                positional_number=positional_number,
            )
            self.prediction_head = Predictior(
                in_dim=self.rxn_attn_model.node_out_dim,
                out_dim=classification_output_dim
            )
        else:
            # If no reaction model, prediction is directly from fused protein features
            self.brige_model = None
            self.interaction_net = None
            self.prediction_head = Predictior(
                in_dim=self.esm_model.output_dim + self.msa_model.msa_out_dim + structure_output_dim,
                out_dim=classification_output_dim
            )



    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:

        graph = batch["protein_graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        esm_output = self.esm_model(graph, graph.node_feature.float())
        node_feature_esm = esm_output["residue_feature"]

        msa_tokens = batch.get('msa_tokens')
        # msa_model 返回填充后的特征和有效掩码
        # msa_features_padded: [B, L_max, msa_output_dim]
        # msa_valid_mask: [B, L_max] (布尔类型, True代表有效残基)

        msa_features_padded, msa_valid_mask= self.msa_model(msa_tokens)
        assert torch.equal(graph.num_residues.to(self._device), msa_valid_mask.sum(dim=1)), f"每个样本的残基数量不匹配，这表明ESM和MSA特征无法安全拼接。"

        # 将有效的MSA特征展平，以对齐可变长度的ESM特征
        # 有效token的数量应等于总残基数 |V_res|
        node_feature_msa = msa_features_padded[msa_valid_mask]
        assert node_feature_esm.shape[0] == node_feature_msa.shape[0], f"ESM ({node_feature_esm.shape[0]}) 和 MSA ({node_feature_msa.shape[0]}) 特征的残基数量不匹配。"
        
        # 形状: (|V_res|, esm_output_dim + msa_output_dim)
        concatenated_features = torch.cat([node_feature_esm, node_feature_msa], dim=-1)

        # 将拼接后的特征送入GearNet. node_feature_struct 形状: (|V_res|, structure_output_dim)
        structure_output = self.structure_model(graph, concatenated_features)
        node_feature_struct = structure_output.get("node_feature")

        protein_features = torch.cat([concatenated_features, node_feature_struct], dim=-1)

        # reaction
        if self.rxn_attn_model:
            output_reaction = self.rxn_attn_model(
                **batch["reaction_graph"],
                return_rts=True
            )
            substrate_node_feature, substrate_mask = output_reaction
            bridged_protein_feature = self.brige_model(protein_features)
            protein_feature_padded, protein_mask = pack_residue_feats(graph, bridged_protein_feature)

            # Cross-Attention
            final_feature_padded, _, _, _, _ = self.interaction_net(
                src=protein_feature_padded,
                tgt=substrate_node_feature,
                src_mask=protein_mask,
                tgt_mask=substrate_mask
            )
            final_feature = final_feature_padded[protein_mask.bool()]
        else:
            final_feature = protein_features
            _, protein_mask = pack_residue_feats(graph, final_feature)

        logits = self.prediction_head(final_feature)

        return logits, protein_mask


    def _load_rxn_attn_model(self, model_state_path, from_scratch):
        if not model_state_path:
            return None
            
        model_state, rxn_attn_args = read_model_state(model_save_path=model_state_path)
        rxn_attn_model = ReactionMGMTurnNet(
            node_in_feats=rxn_attn_args["node_in_feats"],
            node_out_feats=rxn_attn_args["node_out_feats"],
            edge_in_feats=rxn_attn_args["edge_in_feats"],
            edge_hidden_feats=rxn_attn_args["edge_hidden_feats"],
            num_step_message_passing=rxn_attn_args["num_step_message_passing"],
            attention_heads=rxn_attn_args["attention_heads"],
            attention_layers=rxn_attn_args["attention_layers"],
            dropout=rxn_attn_args["dropout"],
            num_atom_types=rxn_attn_args["num_atom_types"],
            cross_attn_h_rate=rxn_attn_args["cross_attn_h_rate"],
            is_pretrain=False,
        )
        if not from_scratch:
            print(
                f"Loading reaction attention model checkpoint from {model_state_path}..."
            )
            rxn_attn_model = load_pretrain_model_state(rxn_attn_model, model_state)
        return rxn_attn_model



class RFSite(nn.Module):

    MODEL_PARAM ={
        "n_module"     : 8,
        "n_module_str" : 4,
        "n_layer"      : 1,
        "d_msa"        : 384,
        "d_pair"       : 288,
        "d_templ"      : 64,
        "n_head_msa"   : 12,
        "n_head_pair"  : 8,
        "n_head_templ" : 4,
        "d_hidden"     : 64,
        "r_ff"         : 4,
        "n_resblock"   : 1,
        "p_drop"       : 0.1,
        "use_templ"    : True,
        "performer_N_opts": {"nb_features": 64},
        "performer_L_opts": {"nb_features": 64}
        }
    SE3_param = {
            "num_layers"    : 2,
            "num_channels"  : 16,
            "num_degrees"   : 2,
            "l0_in_features": 32,
            "l0_out_features": 8,
            "l1_in_features": 3,
            "l1_out_features": 3,
            "num_edge_features": 32,
            "div": 2,
            "n_heads": 4
            }
    MODEL_PARAM['SE3_param'] = SE3_param



    def __init__(self, conf: DictConfig):
        super(RFSite, self).__init__()
        self.conf = conf
        self.predictor_type = self.conf.predictor_type
        self.rosetta_model = self.load_rosettafold(self.conf.rosetta_model_path)
        
        self.d_msa = self.MODEL_PARAM['d_msa']
        self.d_pair = self.MODEL_PARAM['d_pair']
        self.d_state = self.MODEL_PARAM['SE3_param']['l0_out_features']

        # =========================================== Top@Custom ====================================================== 
        # Sequence Feature Extractor
        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=self.conf.esm_model_name)
        self.esm_output_dim = self.esm_model.output_dim
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # Graph Construction
        if self.conf.use_graph_construction:
            self.graph_construction_model = layers.geometry.GraphConstruction(
                node_layers=[layers.geometry.AlphaCarbonNode()],
                edge_layers=[
                    layers.geometry.SequentialEdge(max_distance=2),
                    layers.geometry.SpatialEdge(radius=10, min_distance=5),
                    layers.geometry.KNNEdge(k=10, max_distance=0),
                ],
                edge_feature="gearnet"
            )
        else:
            self.graph_construction_model = None

        # Structure Feature Extractor
        if self.conf.use_gearnet:
            self.structure_model = StructureEncoder(
                input_dim=self.esm_output_dim,
                hidden_dims=[512, 512, 512, 512, 512, 512],
                num_relation=7,
                edge_input_dim=None,
                num_angle_bin=None,
                batch_norm=True,
                short_cut=True,
                readout="sum",
            )
            structure_output_dim = 512
        else:
            self.structure_model = None
            structure_output_dim = 0

        # Reaction Feature Extractor
        if self.conf.rxn_model_path:
            self.rxn_attn_model = self.load_rxn_attn_model(self.conf.rxn_model_path, False)
        else:
            self.rxn_attn_model = None

        # fusion
        protein_feature_dim = self.d_msa + self.d_state + structure_output_dim + self.esm_output_dim
        if self.rxn_attn_model:
            self.brige_model = Predictior(
                in_dim=protein_feature_dim,
                out_dim=self.rxn_attn_model.node_out_dim
            )
            self.interaction_net = GlobalMultiHeadAttention(
                d_model=self.rxn_attn_model.node_out_dim,
                heads=8,
                n_layers=3,
                cross_attn_h_rate=1.0,
                dropout=0.1 if self.training else 0.0,
                positional_number=0,
            )
            if self.predictor_type == "Predictior":
                self.prediction_head = Predictior(
                    in_dim=self.rxn_attn_model.node_out_dim,
                    out_dim=4
                )
            elif self.predictor_type == "PairBiasedAttentionPredictor":
                self.prediction_head = PairBiasedAttentionPredictor(
                    node_dim=self.rxn_attn_model.node_out_dim,
                    pair_dim=self.d_pair,
                    hidden_dim=self.rxn_attn_model.node_out_dim * 2,
                    output_dim=4, # 分类数
                    num_layers=3,
                    num_heads=4,
                    dropout=0.1 if self.training else 0.0
                )
        else:
            self.brige_model = None
            self.interaction_net = None
            if self.predictor_type == "Predictior":
                self.prediction_head = Predictior(
                    in_dim=protein_feature_dim,
                    out_dim=4
                )
            elif self.predictor_type == "PairBiasedAttentionPredictor":
                self.prediction_head = PairBiasedAttentionPredictor(
                    node_dim=protein_feature_dim,
                    pair_dim=self.d_pair,
                    hidden_dim=protein_feature_dim,
                    output_dim=4, # 分类数
                    num_layers=3,
                    num_heads=4,
                    dropout=0.1 if self.training else 0.0
                )



    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        # 步骤 1: 从批次数据中提取必要的输入
        msa = batch['msa']
        seq = batch['seq']
        idx = batch['pdb_idx']
        xyz_27 = batch['xyz_27']
        seq_mask = batch['seq_mask']
        msa_mask = batch['msa_mask']
        B, L, _, _ = xyz_27.shape
        protein_features_list = []

        xyz_t = xyz_27[:, :, :3, :] # 形状: (B, L, 3, 3)
        t1d = torch.zeros((B, 1, L, 3), device=msa.device) # t1d 代表位置比对得分，在此场景下无意义，设为0
        t0d = torch.ones((B, 1, 3), device=msa.device) # t0d 代表模板的全局统计信息 (概率, 序列一致性, 相似度)。因为使用自身结构作为完美模板，我们使用高置信度的值 [1.0, 1.0, 1.0]。
        t2d = rosetta_xyz_to_t2d(xyz_t.unsqueeze(1), t0d, seq_mask)
        
        # msa_feat: (B, L, d_msa), pair_feat: (B, L, L, d_pair), state_feat: (B, L, d_state)
        msa_feat, pair_feat, state_feat = self.rosetta_model(
            msa=msa,
            seq=seq,
            idx=idx,
            xyz=xyz_t,
            seq_mask=seq_mask,
            msa_mask=msa_mask,
            t1d=t1d,
            t2d=t2d
        )
        # =========================================== Top@Custom ====================================================== 
        graph = batch["protein_graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        
        # ESM feature
        esm_output = self.esm_model(graph, graph.node_feature.float())
        esm_features_unpadded = esm_output['residue_feature']
        esm_features_padded, protein_mask = pack_residue_feats(graph, esm_features_unpadded)
        assert torch.equal(protein_mask, seq_mask), f"protein_mask and seq_mask are not identical!"
        protein_features_list.append(esm_features_padded)

        # GearNet
        if self.structure_model:
            structure_output = self.structure_model(graph, esm_features_unpadded)
            struct_features_unpadded = structure_output['node_feature']
            struct_features_padded, _ = pack_residue_feats(graph, struct_features_unpadded)
            assert struct_features_padded.shape[1] == msa_feat.shape[1], f"Structure features length ({struct_features_padded.shape[1]}) and MSA features length ({msa_feat.shape[1]}) do not match."
            protein_features_list.append(struct_features_padded)

        assert esm_features_padded.shape[1] == msa_feat.shape[1], f"ESM features length ({esm_features_padded.shape[1]}) and MSA features length ({msa_feat.shape[1]}) do not match."
        protein_features_list.append(msa_feat)
        protein_features_list.append(state_feat)
        protein_features = torch.cat(protein_features_list, dim=-1) # 形状: (B, L, feature_dim)

        # 
        if self.rxn_attn_model:
            protein_features_down = self.brige_model(protein_features)
            substrate_node_feature, substrate_mask = self.rxn_attn_model(**batch['reaction_graph'], return_rts=True)
            final_feature_padded, _, _, _, _ = self.interaction_net(
                src=protein_features_down,
                tgt=substrate_node_feature,
                src_mask=protein_mask,
                tgt_mask=substrate_mask
            )
            if self.predictor_type == "Predictior":
                final_feature = final_feature_padded[protein_mask]
                logits = self.prediction_head(final_feature)
            elif self.predictor_type == "PairBiasedAttentionPredictor":
                logits_padded = self.prediction_head(
                    node_features=final_feature_padded,
                    pair_features=pair_feat,
                    mask=protein_mask
                )
                logits = logits_padded[protein_mask]
        else:
            if self.predictor_type == "Predictior":
                final_feature = protein_features[protein_mask]
                logits = self.prediction_head(final_feature)
            elif self.predictor_type == "PairBiasedAttentionPredictor":
                logits_padded = self.prediction_head(
                    node_features=protein_features,
                    pair_features=pair_feat,
                    mask=protein_mask
                )
                logits = logits_padded[protein_mask]

        return logits, protein_mask



    def load_rosettafold(self, model_path):
        model = RoseTTAFold(**self.MODEL_PARAM)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未在路径 {model_path} 找到预训练的 RoseTTAFold 模型")
            
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        for param in model.parameters():
            param.requires_grad = False
        print("RoseTTAFold loaded")
    
        return model
    

    def load_rxn_attn_model(self, model_state_path, from_scratch):
        if not model_state_path:
            return None
            
        model_state, rxn_attn_args = read_model_state(model_save_path=model_state_path)
        rxn_attn_model = ReactionMGMTurnNet(
            node_in_feats=rxn_attn_args["node_in_feats"],
            node_out_feats=rxn_attn_args["node_out_feats"],
            edge_in_feats=rxn_attn_args["edge_in_feats"],
            edge_hidden_feats=rxn_attn_args["edge_hidden_feats"],
            num_step_message_passing=rxn_attn_args["num_step_message_passing"],
            attention_heads=rxn_attn_args["attention_heads"],
            attention_layers=rxn_attn_args["attention_layers"],
            dropout=rxn_attn_args["dropout"],
            num_atom_types=rxn_attn_args["num_atom_types"],
            cross_attn_h_rate=rxn_attn_args["cross_attn_h_rate"],
            is_pretrain=False,
        )
        if not from_scratch:
            print(
                f"Loading reaction attention model checkpoint from {model_state_path}..."
            )
            rxn_attn_model = load_pretrain_model_state(rxn_attn_model, model_state)
        return rxn_attn_model
    


class ResidueEncoder(nn.Module):
    """
    ResidueEncoder 是 EasIFA 模型去除最后的分类线性层后所保留的部分
    """
    def __init__(self, conf: DictConfig):
        """
        Args:
            conf: config/ClusterSite-Unified.yaml的model部分的ResidueEncoder部分
        """
        super(ResidueEncoder, self).__init__()
        self.conf = conf
        self.num_active_site_type = 4
        self.enzyme_attn_model = EnzymeAttnNetwork(use_graph_construction_model=True)
        self.rxn_attn_model = self._load_rxn_attn_model(self.conf['rxn_model_path'])
        self.brige_model = Predictior(self.enzyme_attn_model.output_dim, self.rxn_attn_model.node_out_dim)
        self.interaction_net = GlobalMultiHeadAttention(
            self.rxn_attn_model.node_out_dim,
            heads=8,
            n_layers=3,
            cross_attn_h_rate=1,
            dropout=0.1,
            positional_number=0,
        )
        self.active_site_type_net = Predictior(self.rxn_attn_model.node_out_dim, self.num_active_site_type)

    
    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        output_protein = self.enzyme_attn_model(batch)
        output_reaction = self.rxn_attn_model(**batch["reaction_graph"], return_rts=True)
        substrate_node_feature, substrate_mask = output_reaction
        protein_node_feature = self.brige_model(output_protein["node_feature"])
        protein_node_feature, protein_mask = pack_residue_feats(batch["protein_graph"], protein_node_feature)
        protein_node_feature, _, _, _, _ = self.interaction_net(
            src=protein_node_feature,
            tgt=substrate_node_feature,
            src_mask=protein_mask,
            tgt_mask=substrate_mask,
        )
        return protein_node_feature, protein_mask

    
    def _load_rxn_attn_model(self, model_state_path):
        model_state, rxn_attn_args = read_model_state(model_save_path=model_state_path)

        rxn_attn_model = ReactionMGMTurnNet(
            node_in_feats=rxn_attn_args["node_in_feats"],
            node_out_feats=rxn_attn_args["node_out_feats"],
            edge_in_feats=rxn_attn_args["edge_in_feats"],
            edge_hidden_feats=rxn_attn_args["edge_hidden_feats"],
            num_step_message_passing=rxn_attn_args["num_step_message_passing"],
            attention_heads=rxn_attn_args["attention_heads"],
            attention_layers=rxn_attn_args["attention_layers"],
            dropout=rxn_attn_args["dropout"],
            num_atom_types=rxn_attn_args["num_atom_types"],
            cross_attn_h_rate=rxn_attn_args["cross_attn_h_rate"],
            is_pretrain=False,
        )
        rxn_attn_model = load_pretrain_model_state(rxn_attn_model, model_state)

        return rxn_attn_model



class ECEmbeddingNet(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, 
                 output_dim: int = 128, dropout: float = 0.1):
        super(ECEmbeddingNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.dropout(self.ln1(self.fc1(x)))
        x = F.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = F.relu(x)
        x = self.fc3(x)
        return x



class StackingC(nn.Module):
    """
    模块三：每个类别 c 独立做一维线性回归（输入是两路概率 p_ce_c 与 p_ref_c）
    形式： \hat{y}_c = w_c1 * p_ce_c + w_c2 * p_ref_c + b_c
    """
    def __init__(self, num_class: int = 4, temperature: float = 1.0):
        super().__init__()
        self.num_class = num_class
        self.temperature = float(temperature)
        # 参数：每类一组权重（2维）与偏置
        self.w = nn.Parameter(torch.ones(num_class, 2) * 0.5)  # 初值 0.5/0.5
        self.b = nn.Parameter(torch.zeros(num_class))

    def forward(self, p_ce: torch.Tensor, p_ref: torch.Tensor):
        """
        [
        \text{logits}[n,c] = w_{c1} p_{\text{ce}}[n,c] + w_{c2} p_{\text{ref}}[n,c] + b_c
        ]
        
        Args:
            p_ce  : [N, C]，模块一 softmax 概率 (扁平化)
            p_ref : [N, C]，模块二 概率 (扁平化)
        Returns:
            probs : [N, C] 融合后的概率
            logits: [N, C] 未归一化分数
        """
        assert p_ce.dim() == 2 and p_ref.dim() == 2, f"Expected flat inputs of shape [N, C], but got {p_ce.shape} and {p_ref.shape}"
        
        # x: [N, C, 2]
        x = torch.stack([p_ce, p_ref], dim=-1)
        
        # 广播做逐类点积：[N,C,2]·[C,2] -> [N,C]
        # self.w.view(1, self.num_class, 2) -> [1, C, 2]
        # self.b.view(1, self.num_class)     -> [1, C]
        logits = (x * self.w.view(1, self.num_class, 2)).sum(-1) + self.b.view(1, self.num_class)
        
        if self.temperature is not None and self.temperature > 0:
            logits = logits / self.temperature
            
        probs = torch.softmax(logits, dim=-1)
        return probs, logits



# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
class SupConProjectionHead(nn.Module):
    """
    Supervised Contrastive Learning 风格的投影头:
    - 两层全连接，第一层带 ReLU，第二层无 bias (遵循 SupCon 官方实现)
    - 输出做 L2 归一化
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SupConProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] 残基级表示 h_i
        Returns:
            z: [N, output_dim], 已在最后一维做 L2 归一化
        """
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = F.normalize(x, dim=-1)
        return x



class ResidueEncoderAlpha(nn.Module):
    """
    ResidueEncoderAlpha: 结构消融版本的残基编码器
    $$ RXNFP模型必须在cuda:0上运行
    """
    
    def __init__(self, config: DictConfig):
        super(ResidueEncoderAlpha, self).__init__()
        self.config = config
        
        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=self.config['esm_model_name'])
        self.esm_output_dim = self.esm_model.output_dim
        self.brige_model = Predictior(self.esm_output_dim, self.config['bridge_output_dim'])
        
        self.rxnfp_model, self.rxnfp_tokenizer = get_default_model_and_tokenizer()
        self.rxnfp_model.eval()
        for p in self.rxnfp_model.parameters():
            p.requires_grad_(False)
        self.rxnfp_generator = RXNBERTFingerprintGenerator(self.rxnfp_model, self.rxnfp_tokenizer)
        assert self.rxnfp_model.config.hidden_size == self.config['rxnfp_output_dim'], (f"RXNFP hidden_size={self.rxnfp_model.config.hidden_size} "f"!= config rxnfp_output_dim={self.config['rxnfp_output_dim']}")
        self.rxn_projector = nn.Sequential(
            nn.Linear(self.config['rxnfp_output_dim'], self.config['bridge_output_dim']),
            nn.ReLU(),
            nn.Linear(self.config['bridge_output_dim'], self.config['bridge_output_dim'])
        )
        
        self.rgmm_layers = nn.ModuleList([
            RGMM(
                dim_model=self.config['bridge_output_dim'],
                init_k_size=self.config['init_k_size'],
                depth=self.config['pka_depth'],
                scale_factor=self.config['scale_factor']
            )
            for _ in range(self.config['rgmm_num_layers'])
        ])

        self.prediction_head = Predictior(self.config['bridge_output_dim'], self.config['num_class'])
        self.proj_head = SupConProjectionHead(input_dim=self.config['bridge_output_dim'], hidden_dim=self.config['hidden_dim'], output_dim=self.config['output_dim'])
        self.rxnfp_cache: Dict[str, np.ndarray] = {}
    

    
    def forward(self, batch: Dict[str, Any], return_features: bool = False, return_z: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        graph = batch["protein_graph"]
        output_esm = self.esm_model(graph, graph.node_feature)
        node_feature_seq = output_esm.get("residue_feature")  # [N_total, esm_output_dim]
        
        final_node_feature = self.brige_model(node_feature_seq)  # [N_total, bridge_output_dim]
        padded_feature, protein_mask = pack_residue_feats(graph, final_node_feature)  # [B, L, bridge_output_dim], [B, L]
        
        reaction_emb = self.smiles_to_rxnfp_embedding(batch["canonicalize_rxn_smiles"])  # [B, rxnfp_output_dim]
        rxn_feat = self.rxn_projector(reaction_emb)  # [B, bridge_output_dim]
        
        h_v = padded_feature
        for layer in self.rgmm_layers:
            h_v = layer(h_v, rxn_feat.unsqueeze(1), protein_mask.bool(), None)
        
        if return_features:
            return h_v, protein_mask
        
        h_v_flat = h_v[protein_mask.bool()]
        logits = self.prediction_head(h_v_flat)
        
        if return_z:
            z = self.proj_head(h_v_flat)
            return logits, z, protein_mask
        
        return logits, protein_mask
    
    
    @torch.no_grad()
    def smiles_to_rxnfp_embedding(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Args: 
            smiles_list: List[str], len=B
        Returns: 
            rxn: [B, rxnfp_output_dim] 的 torch.Tensor
        """
        raw_need = [s for s in smiles_list if (s != "" and s not in self.rxnfp_cache)]
        need = sorted(list(set(raw_need))) 
        if len(need) > 0:
            embs = self.rxnfp_generator.convert_batch(need)  # list[np.array] or list[list]
            check_rxnfp(embs, self.config["rxnfp_output_dim"])
            for s, e in zip(need, embs):
                self.rxnfp_cache[s] = np.asarray(e, dtype=np.float32)

        mat = np.zeros((len(smiles_list), self.config['rxnfp_output_dim']), dtype=np.float32)
        for i, s in enumerate(smiles_list):
            assert s != "" and s in self.rxnfp_cache, f"SMILES '{s}' not in cache after generation or SMILES is None."
            mat[i] = self.rxnfp_cache[s]

        rxn = torch.from_numpy(mat).to(device=(next(self.rxn_projector.parameters())).device, dtype=(next(self.rxn_projector.parameters())).dtype)  # [B, dim]
        return rxn



class ResidueEncoderBeta(nn.Module):
    """
    ResidueEncoderBeta 是 ResidueEncoderGamma 去掉 reaction 部分的消融变体。
    仅保留: ESM + GearNet(结构编码器) + Bridge融合 + 分类头 + SupCon投影头
    移除的模块: RXNFP模型、rxn_projector、RGMM层
    
    接口与 ResidueEncoderGamma 保持一致，支持三种模式：
        - return_features=True: 返回残基特征（用于下游模块）
        - return_features=False, return_z=False: 返回logits（普通CE训练）
        - return_features=False, return_z=True: 返回logits和z（SupCon训练）
    """
    def __init__(self, config: DictConfig):
        """
        Args:
            config: 总配置文件的model部分的ResidueEncoderBeta部分
        """
        super(ResidueEncoderBeta, self).__init__()
        self.config = config

        self.graph_construction_model = layers.geometry.GraphConstruction(
            node_layers=[layers.geometry.AlphaCarbonNode()],
            edge_layers=[
                layers.geometry.SequentialEdge(max_distance=2),
                layers.geometry.SpatialEdge(radius=10, min_distance=5),
                layers.geometry.KNNEdge(k=10, max_distance=0),
            ],
            edge_feature="gearnet"
        )

        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=self.config['esm_model_name'])
        self.esm_output_dim = self.esm_model.output_dim

        self.structure_model = StructureEncoder(
            input_dim=self.esm_output_dim,
            hidden_dims=self.config['structure_hidden_dims'],
            num_relation=self.config['num_relation'],
            edge_input_dim=self.config['edge_input_dim'],
            num_angle_bin=self.config['num_angle_bin'],
            batch_norm=self.config['structure_batch_norm'],
            short_cut=self.config['short_cut'],
            readout=self.config['readout'],
        )

        self.brige_model = Predictior(self.esm_output_dim + self.config['structure_hidden_dims'][-1], self.config['bridge_output_dim'])
        
        # =================================================================================================
        self.prediction_head = Predictior(self.config['bridge_output_dim'], self.config['num_class'])
        self.proj_head = SupConProjectionHead(input_dim=self.config['bridge_output_dim'], hidden_dim=self.config['hidden_dim'], output_dim=self.config['output_dim'])


    def forward(self, batch: Dict[str, Any], return_features: bool = False, return_z: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，接口与 ResidueEncoderGamma 保持一致。
        
        Args:
            batch: 输入batch，包含 'protein_graph' 等字段
            return_features: 是否返回残基特征（用于模块二的训练和推理）
            return_z: 是否返回SupCon投影向量z（用于对比学习训练）
        
        Returns:
            - return_features=True: (padded_feature, protein_mask)  模式1: 仅提取特征
            - return_features=False, return_z=False: (logits, protein_mask)  模式2: 普通CE训练
            - return_features=False, return_z=True: (logits, z, protein_mask)  模式3: SupCon训练
        """
        graph = batch["protein_graph"]
        graph = self.graph_construction_model(graph)
        output_esm = self.esm_model(graph, graph.node_feature)
        node_feature_seq = output_esm.get("residue_feature")

        output_struct = self.structure_model(graph, node_feature_seq)
        node_feature_struct = output_struct.get("node_feature", output_struct.get("residue_feature"))

        concat_node_feature = torch.cat([node_feature_seq, node_feature_struct], dim=-1)
        final_node_feature = self.brige_model(concat_node_feature)  # [N_total, D_bridge]
        padded_feature, protein_mask = pack_residue_feats(graph, final_node_feature)  # h_v: [B, L, D]

        # 注意：ResidueEncoderBeta 没有 RGMM 层，所以 h_v 直接等于 padded_feature
        h_v = padded_feature

        if return_features:
            return h_v, protein_mask            # 模式 1: 仅提取特征

        h_v_flat = h_v[protein_mask.bool()]
        logits = self.prediction_head(h_v_flat)

        if return_z:
            z = self.proj_head(h_v_flat)
            return logits, z, protein_mask      # 模式 3: SupCon 训练模式
        
        return logits, protein_mask             # 模式 2: 普通 CE 训练模式



class PKA(nn.Module):
    """
    Parallel Multi-Scale PKA (无反应条件路由)
    - 并联多尺度 depthwise Conv1d：不同 kernel_size + dilation 覆盖不同序列跨度
    - concat 后用 1x1 pointwise 融合回 dim_model（更“并联”、更像多尺度金字塔）
    - mask-aware：pad 位置不输出；可选 mask-renorm 减轻边界 pad 对邻近 valid 的影响
    """
    def __init__(
        self,
        dim_model=128,
        init_k_size=3,
        depth=4,
        scale_factor=4,
        use_dilation=True,
        dilation_growth=2,      # dilation: 1,2,4,8...
        use_mask_renorm=True,   # 推荐 True：更稳
        dropout=0.0,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.depth = depth
        self.use_mask_renorm = use_mask_renorm
        self.dropout = nn.Dropout(dropout)

        self.ks = []
        self.ds = []
        self.branches = nn.ModuleList()

        for i in range(depth):
            k = init_k_size + i * scale_factor
            if k % 2 == 0:
                k += 1
            d = (dilation_growth ** i) if use_dilation else 1
            pad = (k // 2) * d

            self.ks.append(k)
            self.ds.append(d)

            # depthwise conv：每个通道独立，多尺度、计算量低
            self.branches.append(
                nn.Conv1d(
                    dim_model, dim_model,
                    kernel_size=k,
                    dilation=d,
                    padding=pad,
                    groups=dim_model,
                    bias=False
                )
            )

        self.act = nn.GELU()

        # concat 融合：把 depth 个分支拼到通道维，再用 1x1 压回 dim_model
        self.fuse = nn.Sequential(
            nn.Conv1d(dim_model * depth, dim_model, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.out_ffn = nn.Linear(dim_model, dim_model)

    def forward(self, x, valid_mask=None):
        """
        x: [B, L, D]
        valid_mask: [B, L]  True=valid, False=pad
        """
        B, L, D = x.shape
        assert D == self.dim_model

        if valid_mask is None:
            valid_mask = torch.ones((B, L), device=x.device, dtype=torch.bool)

        xt = x.transpose(1, 2)  # [B, D, L]
        v = valid_mask.to(dtype=xt.dtype, device=xt.device).unsqueeze(1)  # [B,1,L]
        xt = xt * v  # pad 清零

        outs = []
        for conv, k, d in zip(self.branches, self.ks, self.ds):
            yi = conv(xt)  # [B,D,L]

            if self.use_mask_renorm:
                # 用同样的 (k,d,pad) 统计每个位置窗口内 valid 数，做归一化，减轻边界 pad 的“稀释”
                ones = torch.ones((1, 1, k), device=xt.device, dtype=xt.dtype)
                denom = F.conv1d(v, ones, padding=(k // 2) * d, dilation=d)  # [B,1,L]
                yi = yi / denom.clamp(min=1.0)

            yi = yi * v
            yi = self.act(yi)
            outs.append(yi)

        y = torch.cat(outs, dim=1)      # [B, D*depth, L]
        y = self.dropout(y)
        y = self.fuse(y)                # [B, D, L]
        y = y.transpose(1, 2)           # [B, L, D]
        y = self.out_ffn(y)
        y = y * valid_mask.to(dtype=y.dtype, device=y.device).unsqueeze(-1)
        return y



class RGMM(nn.Module):
    def __init__(self, dim_model=128, init_k_size=5, depth=4, scale_factor=4):
        super(RGMM, self).__init__()
        self.dim_model = dim_model
        
        self.pka = PKA(dim_model=dim_model, init_k_size=init_k_size, depth=depth, scale_factor=scale_factor)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_model*4, dim_model),
            nn.Dropout(0.1)
        )
        # Layer Norms (Elementwise affine is False because we use FiLM modulation)
        self.norm1 = nn.LayerNorm(dim_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim_model, elementwise_affine=False, eps=1e-6)

        # Cross-Attention projection layers
        self.w_reaction = nn.Linear(dim_model, dim_model, bias=False) # Project Reaction
        self.w_residue = nn.Linear(dim_model, dim_model, bias=False)  # Project Residue
        
        # FiLM Generator: 从 Context Summary 生成调制参数
        self.adaLN_modulation = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_model, 4 * dim_model) # 生成 shift_pka, scale_pka, shift_ffn, scale_ffn
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, residue_feat, reaction_feat, residue_mask, reaction_mask=None):
        """
        Args:
            residue_feat: [B, L_res, D] 蛋白残基特征
            reaction_feat: [B, 1, D]    反应特征 (RxnFP CLS向量)
            residue_mask: [B, L_res]    蛋白padding mask (False为padding)
            reaction_mask: [Ignored]    因为L_rxn=1，不需要mask
        """
        # ================================================================================================= 
        # 1. 计算 Cross-Attention 得到 Condition Summary (c)
        reaction_proj = self.w_reaction(reaction_feat)          # reaction_proj: [B, 1, D] -> 作为 Global Query
        residue_proj = self.w_residue(residue_feat)             # residue_proj: [B, L_res, D] -> 作为 Key & Value
        
        # 计算注意力分数: <Residue_i, Reaction>
        # [B, L_res, D] @ [B, D, 1] -> [B, L_res, 1]
        # 这一步计算每个残基与当前反应的相关性
        attn_scores = torch.bmm(residue_proj, reaction_proj.transpose(1, 2)) / math.sqrt(self.dim_model)
        
        # Mask padding residues (屏蔽掉填充的残基，不让它们贡献到 context c 中)
        if residue_mask is not None:
            attn_scores = attn_scores.masked_fill(~residue_mask.unsqueeze(-1), -1e9) # residue_mask is [B, L_res], unsqueeze to [B, L_res, 1]
        
        # Softmax over Protein Sequence dimension (L_res)，得到归一化的权重: 哪些残基对这个反应最重要
        attn_weights = F.softmax(attn_scores, dim=1) # [B, L_res, 1]
        
        # 加权求和得到 Context Summary
        # [B, L_res, D] * [B, L_res, 1] -> [B, L_res, D] -> sum -> [B, D]
        # 这个 c 代表了"在这个反应语境下，整个蛋白的关键特征概览"
        context_summary = (residue_proj * attn_weights).sum(dim=1) 


        # ================================================================================================= 
        # 2. 生成 FiLM 参数
        # Split output into 4 parts: shift/scale for PKA block, shift/scale for FFN block
        shift_pka, scale_pka, shift_ffn, scale_ffn = self.adaLN_modulation(context_summary).chunk(4, dim=1)

        # =================================================================================================
        # 3. 调制与前向 (Residue Update)
        x = residue_feat
        
        # Block 1: PKA (Poly Kernel Attention)
        # 流程: Norm -> FiLM Modulate -> PKA -> Residual Add
        x_norm1 = self.norm1(x)
        x_modulated1 = x_norm1 * (1 + scale_pka.unsqueeze(1)) + shift_pka.unsqueeze(1)
        x = x + self.pka(x_modulated1, residue_mask)
        x = x * (residue_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1))

        # Block 2: FFN (Feed Forward Network)
        # 流程: Norm -> FiLM Modulate -> MLP -> Residual Add
        x_norm2 = self.norm2(x)
        x_modulated2 = x_norm2 * (1 + scale_ffn.unsqueeze(1)) + shift_ffn.unsqueeze(1)
        x = x + self.ffn(x_modulated2)
        x = x * (residue_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1))

        return x



class ResidueEncoderGamma(nn.Module):
    def __init__(self, config: DictConfig):
        """
        $$ 注意RXNBERTFingerprintGenerator必须在cuda:0上运行
        Args:
            config: 总配置文件的model部分的ResidueEncoderGamma部分
        """
        super(ResidueEncoderGamma, self).__init__()
        self.config = config
        
        self.graph_construction_model = layers.geometry.GraphConstruction(
            node_layers=[layers.geometry.AlphaCarbonNode()],
            edge_layers=[
                layers.geometry.SequentialEdge(max_distance=2),
                layers.geometry.SpatialEdge(radius=10, min_distance=5),
                layers.geometry.KNNEdge(k=10, max_distance=0),
            ],
            edge_feature="gearnet"
        )

        # =================================================================================================
        self.esm_model = EvolutionaryScaleModelingBeta(path="~/.cache/torch/hub/checkpoints", model=self.config['esm_model_name'])
        self.esm_output_dim = self.esm_model.output_dim

        self.structure_model = StructureEncoder(
            input_dim=self.esm_output_dim,
            hidden_dims=self.config['structure_hidden_dims'],
            num_relation=self.config['num_relation'],
            edge_input_dim=self.config['edge_input_dim'],
            num_angle_bin=self.config['num_angle_bin'],
            batch_norm=self.config['structure_batch_norm'],
            short_cut=self.config['short_cut'],
            readout=self.config['readout'],
        )

        self.brige_model = Predictior(self.esm_output_dim + self.config['structure_hidden_dims'][-1], self.config['bridge_output_dim'])


        # =================================================================================================
        self.rxnfp_model, self.rxnfp_tokenizer = get_default_model_and_tokenizer()
        self.rxnfp_model.eval()
        for p in self.rxnfp_model.parameters():
            p.requires_grad_(False)
        self.rxnfp_generator = RXNBERTFingerprintGenerator(self.rxnfp_model, self.rxnfp_tokenizer)
        assert self.rxnfp_model.config.hidden_size == self.config['rxnfp_output_dim'], (f"RXNFP hidden_size={self.rxnfp_model.config.hidden_size} "f"!= config rxnfp_output_dim={self.config['rxnfp_output_dim']}")

        self.rxn_projector = nn.Sequential(
            nn.Linear(self.config['rxnfp_output_dim'], self.config['bridge_output_dim']),
            nn.ReLU(),
            nn.Linear(self.config['bridge_output_dim'], self.config['bridge_output_dim'])
        )
        

        # =================================================================================================
        self.rgmm_layers = nn.ModuleList([
            RGMM(
                dim_model=self.config['bridge_output_dim'],
                init_k_size=self.config['init_k_size'],
                depth=self.config['pka_depth'],
                scale_factor=self.config['scale_factor']
            )
            for _ in range(self.config['rgmm_num_layers'])
        ])


        # =================================================================================================
        self.prediction_head = Predictior(self.config['bridge_output_dim'], self.config['num_class'])
        self.proj_head = SupConProjectionHead(input_dim=self.config['bridge_output_dim'], hidden_dim=self.config['hidden_dim'], output_dim=self.config['output_dim'])
        self.rxnfp_cache: Dict[str, np.ndarray] = {}



    def forward(self, batch: Dict[str, Any], return_features: bool = False, return_z: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        graph = batch["protein_graph"]
        graph = self.graph_construction_model(graph)
        output_esm = self.esm_model(graph, graph.node_feature)
        node_feature_seq = output_esm.get("residue_feature")

        output_struct = self.structure_model(graph, node_feature_seq)
        node_feature_struct = output_struct.get("node_feature", output_struct.get("residue_feature"))

        concat_node_feature = torch.cat([node_feature_seq, node_feature_struct], dim=-1)
        final_node_feature = self.brige_model(concat_node_feature) # [N_total, D_bridge]
        padded_feature, protein_mask = pack_residue_feats(graph, final_node_feature) # h_v: [B, L, D]

        reaction_emb = self.smiles_to_rxnfp_embedding(batch["canonicalize_rxn_smiles"])
        rxn_feat = self.rxn_projector(reaction_emb)

        h_v = padded_feature
        for layer in self.rgmm_layers:
            h_v = layer(h_v, rxn_feat.unsqueeze(1), protein_mask.bool(), None)  # [B,L,D]

        if return_features:
            return h_v, protein_mask            # 模式 1: 仅提取特征

        h_v_flat = h_v[protein_mask.bool()]
        logits = self.prediction_head(h_v_flat)

        if return_z:
            z = self.proj_head(h_v_flat)
            return logits, z, protein_mask      # 模式 3: SupCon 训练模式
        
        return logits, protein_mask             # 模式 2: 普通 CE 训练模式



    @torch.no_grad()
    def smiles_to_rxnfp_embedding(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Args: 
            smiles_list: List[str], len=B
        Returns: 
            rxn: [B, rxnfp_output_dim] 的 torch.Tensor
        """
        raw_need = [s for s in smiles_list if (s != "" and s not in self.rxnfp_cache)]
        need = sorted(list(set(raw_need))) 
        if len(need) > 0:
            embs = self.rxnfp_generator.convert_batch(need)  # list[np.array] or list[list]
            check_rxnfp(embs, self.config["rxnfp_output_dim"])
            for s, e in zip(need, embs):
                self.rxnfp_cache[s] = np.asarray(e, dtype=np.float32)

        mat = np.zeros((len(smiles_list), self.config['rxnfp_output_dim']), dtype=np.float32)
        for i, s in enumerate(smiles_list):
            assert s != "" and s in self.rxnfp_cache, f"SMILES '{s}' not in cache after generation or SMILES is None."
            mat[i] = self.rxnfp_cache[s]

        rxn = torch.from_numpy(mat).to(device=(next(self.rxn_projector.parameters())).device, dtype=(next(self.rxn_projector.parameters())).dtype)  # [B, dim]
        return rxn
            


class ResidueEncoderGammaBinary(ResidueEncoderGamma):
    def __init__(self, config: DictConfig):
        """
        $$ 注意RXNBERTFingerprintGenerator必须在cuda:0上运行
        Args:
            config: 总配置文件的model部分的ResidueEncoderGamma部分
        """
        if config['num_class'] != 2:
            print(f"ResidueEncoderGammaBinary: config['num_class'] is {config['num_class']}, forcing to 2.")
            config['num_class'] = 2
        super().__init__(config)



class ActiveSiteEmbeddingNet(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, 
                 output_dim: int = 128, dropout: float = 0.1):
        super(ActiveSiteEmbeddingNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.dropout(self.ln1(self.fc1(x)))
        x = F.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = F.relu(x)
        x = self.fc3(x)
        x = F.normalize(x, p=2, dim=-1)
        return x



class ClusterSite(nn.Module):
    def __init__(self, config: dict):
        """
        Args:
            config: 总配置文件的model部分
        """
        super(ClusterSite, self).__init__()
        self.config = config

        self.residue_encoder = ResidueEncoderGamma(config['ResidueEncoderGamma'])
        self.active_site_embedding_net = ActiveSiteEmbeddingNet(
            input_dim=config['ActiveSiteEmbeddingNet']['input_dim'],
            hidden_dim=config['ActiveSiteEmbeddingNet']['hidden_dim'],
            output_dim=config['ActiveSiteEmbeddingNet']['output_dim'],
            dropout=config['ActiveSiteEmbeddingNet']['dropout']
        )

    def set_module_train_mode(self, module_name: str):
        """
        激活选定的模块
        """
        self.eval()
        if hasattr(self, module_name):
            target_module = getattr(self, module_name)
            target_module.train()
            print(f"Set '{module_name}' to TRAIN mode, others to EVAL mode.")
        else:
            raise AttributeError(f"Module '{module_name}' not found in ClusterSite model.")



class ClusterSiteBinary(ClusterSite):
    def __init__(self, config: dict):
        super(ClusterSiteBinary, self).__init__(config)
        # Override residue_encoder with binary version
        self.residue_encoder = ResidueEncoderGammaBinary(config['ResidueEncoderGamma'])



class ClusterSiteAlpha(nn.Module):
    def __init__(self, config: dict):
        super(ClusterSiteAlpha, self).__init__()
        self.config = config
        

        self.residue_encoder = ResidueEncoderAlpha(config['ResidueEncoderAlpha'])
        self.active_site_embedding_net = ActiveSiteEmbeddingNet(
            input_dim=config['ActiveSiteEmbeddingNet']['input_dim'],
            hidden_dim=config['ActiveSiteEmbeddingNet']['hidden_dim'],
            output_dim=config['ActiveSiteEmbeddingNet']['output_dim'],
            dropout=config['ActiveSiteEmbeddingNet']['dropout']
        )
        assert config['ResidueEncoderAlpha']['bridge_output_dim'] == config['ActiveSiteEmbeddingNet']['input_dim'], (
            f"Dimension mismatch: "
            f"ResidueEncoderAlpha.bridge_output_dim={config['ResidueEncoderAlpha']['bridge_output_dim']} "
            f"!= ActiveSiteEmbeddingNet.input_dim={config['ActiveSiteEmbeddingNet']['input_dim']}"
        )
    
    
    def set_module_train_mode(self, module_name: str):
        """
        激活选定的模块
        """
        self.eval()
        if hasattr(self, module_name):
            target_module = getattr(self, module_name)
            target_module.train()
            print(f"Set '{module_name}' to TRAIN mode, others to EVAL mode.")
        else:
            raise AttributeError(f"Module '{module_name}' not found in ClusterSite model.")
        


class ClusterSiteBeta(nn.Module):
    """
    ClusterSiteBeta 是 ClusterSite 的消融变体，用于探究去除 reaction（RGMM模块）对整体的影响。
    
    与 ClusterSite 的区别：
        - 使用 ResidueEncoderBeta 替代 ResidueEncoderGamma
        - ResidueEncoderBeta 没有 RXNFP 模型、rxn_projector 和 RGMM 层
    
    模块组成：
        - residue_encoder: ResidueEncoderBeta（无反应信息的残基编码器）
        - active_site_embedding_net: ActiveSiteEmbeddingNet（与ClusterSite相同）
    """
    def __init__(self, config: dict):
        """
        Args:
            config: 总配置文件的model部分，需要包含 'ResidueEncoderBeta' 和 'ActiveSiteEmbeddingNet' 两个子配置
        """
        super(ClusterSiteBeta, self).__init__()
        self.config = config

        # 使用 ResidueEncoderBeta 替代 ResidueEncoderGamma（无反应信息）
        self.residue_encoder = ResidueEncoderBeta(config['ResidueEncoderBeta'])
        self.active_site_embedding_net = ActiveSiteEmbeddingNet(
            input_dim=config['ActiveSiteEmbeddingNet']['input_dim'],
            hidden_dim=config['ActiveSiteEmbeddingNet']['hidden_dim'],
            output_dim=config['ActiveSiteEmbeddingNet']['output_dim'],
            dropout=config['ActiveSiteEmbeddingNet']['dropout']
        )

    def set_module_train_mode(self, module_name: str):
        """
        激活选定的模块，其他模块设为 eval 模式。
        
        Args:
            module_name: 模块名称，可选 'residue_encoder' 或 'active_site_embedding_net'
        """
        self.eval()
        if hasattr(self, module_name):
            target_module = getattr(self, module_name)
            target_module.train()
            print(f"Set '{module_name}' to TRAIN mode, others to EVAL mode.")
        else:
            raise AttributeError(f"Module '{module_name}' not found in ClusterSiteBeta model.")




