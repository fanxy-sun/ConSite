import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time
import random
import os
import math
from collections import defaultdict
import esm
from esm.modules import ESM1bLayerNorm, RobertaLMHead, TransformerLayer
from esm.multihead_attention import MultiheadAttention, utils_softmax

from data_loaders.enzyme_msa_dataloaders import MatrixLinearNormalizer


class DCABiasedMultiheadAttention(MultiheadAttention):
    """
    继承自ESM的MultiheadAttention，添加DCA偏置注入功能
    在softmax之前注入DCA耦合矩阵偏置，实现：Attention(Q,K,V) = softmax(QK^T/√d_k + B_DCA)V，其中B_DCA是来自DCA的耦合矩阵偏置，训练过程中冻结不变，仅优化Transformer主干参数。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dca_bias_matrices = None  
        self._bias_scaling_factor = 1.0

        # 默认设置，会在替换时从原模块或alphabet中覆盖
        self.prepend_bos = True
        self.append_eos = True



    def set_dca_bias(self, bias_matrices: List[torch.Tensor], scaling_factor: float = 1.0):
        """
        设置一批DCA偏置矩阵，支持每个序列有不同的长度
        
        Args:
            bias_matrices: 偏置矩阵列表，每个[valid_len_i, valid_len_i]
            scaling_factor: 偏置缩放因子，控制DCA偏置对注意力的影响强度
        """
        self._dca_bias_matrices = bias_matrices
        self._bias_scaling_factor = scaling_factor
        

        
    def clear_dca_bias(self, preserve_scaling: bool = False):
        """清除DCA偏置矩阵"""
        self._dca_bias_matrices = None  # 确保清除复数形式属性
        if not preserve_scaling:
            self._bias_scaling_factor = 1.0
    


    def prepare_dca_bias(self, attn_weights: torch.Tensor) -> Optional[torch.Tensor]:
        """
        准备偏置矩阵以匹配注意力权重的形状：对耦合矩阵填充至seq_len，并且乘以scaling_factor

        使用 torch.stack 和 F.pad 替代 for 循环中的原地赋值，以兼容 torch.compile
        
        Args:
            attn_weights: 注意力权重 [bsz * num_heads, seq_len, seq_len]
                
        Returns:
            准备好的偏置矩阵或None
        """

        if not hasattr(self, '_dca_bias_matrices') or self._dca_bias_matrices is None:
            return None
            
        bias_matrices = self._dca_bias_matrices
        bsz_heads, seq_len, _ = attn_weights.shape
        bsz = bsz_heads // self.num_heads
        assert bsz == len(bias_matrices)
        device = attn_weights.device
        dtype = attn_weights.dtype
        content_start = 1 if self.prepend_bos else 0
        
        # 创建填充后的矩阵列表
        padded_matrices = []
        for matrix in bias_matrices:
            valid_len = matrix.size(0)
            if content_start + valid_len > seq_len:
                raise ValueError(f"content_start + valid_len > seq_len: {content_start + valid_len} > {seq_len}")
            
            # 创建目标大小的全零张量并填充
            padded_matrix = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
            if valid_len > 0:
                end_pos = content_start + valid_len
                padded_matrix[content_start:end_pos, content_start:end_pos] = matrix.to(device=device, dtype=dtype)
            padded_matrices.append(padded_matrix)

        # 将列表堆叠成一个批处理张量 [bsz, seq_len, seq_len]
        if padded_matrices:
            bias_tensor = torch.stack(padded_matrices, dim=0)
        else:
            raise ValueError("bias_matrices is empty")
        
        # 扩展到多头维度以匹配 attn_weights 的形状
        # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
        bias_tensor = bias_tensor.unsqueeze(1)
        # [bsz, 1, seq_len, seq_len] -> [bsz, num_heads, seq_len, seq_len]
        bias_tensor = bias_tensor.repeat(1, self.num_heads, 1, 1)
        # [bsz, num_heads, seq_len, seq_len] -> [bsz * num_heads, seq_len, seq_len]
        bias_tensor = bias_tensor.reshape(bsz * self.num_heads, seq_len, seq_len)
        
        return bias_tensor * self._bias_scaling_factor



    def forward(
        self,
        query,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward function for DCABiasedMultiheadAttention.
        Implements the formula: Attention(Q,K,V) = softmax(QK^T/√d_k + B_DCA)V
        
        Args:
            query: Query tensor, [tgt_len, bsz, embed_dim]
            key: Key tensor (optional), [src_len, bsz, embed_dim]
            value: Value tensor (optional), [src_len, bsz, embed_dim]
            key_padding_mask: Mask for keys that are padding (optional), 键填充掩码[bsz, src_len]
            incremental_state: State for incremental decoding (optional)
            need_weights: Whether to return attention weights
            static_kv: Whether key and value are static
            attn_mask: Mask applied to attention weights (optional), 注意力掩码[tgt_len, src_len]
            before_softmax: Whether to return attention weights before softmax
            need_head_weights: Whether to return attention weights for each head
            
        Returns:
            Tuple of output tensor and optional attention weights
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # 如果可以使用PyTorch的优化版本且没有DCA偏置，使用原始实现
        if (
            not self.rot_emb
            and self.enable_torch_version
            and not self.onnx_trace
            and incremental_state is None
            and not static_kv
            and self._dca_bias_matrices is None  # 没有DCA偏置时才使用优化版本
            and not need_head_weights
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        # 处理增量状态
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
            else:
                saved_state = None
        else:
            saved_state = None


        if self.self_attention:
            q = self.q_proj(query)      # [tgt_len, bsz, embed_dim]
            k = self.k_proj(query)      # [tgt_len, bsz, embed_dim]
            v = self.v_proj(query)      # [tgt_len, bsz, embed_dim]
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = q * self.scaling        # [tgt_len, bsz, embed_dim], scaling = head_dim^(-0.5)

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        # 重塑为多头格式
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)       # q: [bsz * num_heads, tgt_len, head_dim]
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)        # k: [bsz * num_heads, src_len, head_dim]
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)        # v: [bsz * num_heads, src_len, head_dim]

        # 完整的增量状态处理逻辑
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        # 处理zero attention
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        # 应用旋转位置编码
        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        # ============= 关键部分：计算注意力权重 =============
        attn_weights = torch.bmm(q, k.transpose(1, 2))      # attn_weights: [bsz * num_heads, tgt_len, src_len]
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # 应用注意力掩码
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)      # [1, tgt_len, src_len]
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights = attn_weights + attn_mask       # [bsz * num_heads, tgt_len, src_len]

        # 应用key padding掩码
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # =========================================== Top@关键修改：在softmax之前注入DCA偏置 ======================================================
        # dca_bias: [bsz * num_heads, tgt_len, src_len] 或 None
        # 在bias_injection情境下，一般是自注意力 tgt_len == src_len，所以dca_bias: [bsz * num_heads, tgt_len, tgt_len] 或 None
        dca_bias = self.prepare_dca_bias(attn_weights)      
        if dca_bias is not None:
            attn_weights = attn_weights + dca_bias      # attn_weights: [bsz * num_heads, tgt_len, tgt_len]

        # 如果需要返回softmax之前的权重
        if before_softmax:
            return attn_weights, v

        # 应用softmax
        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)        # attn_weights_float: [bsz * num_heads, tgt_len, tgt_len]
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(                 # attn_probs: [bsz * num_heads, tgt_len, tgt_len]
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )

        # 计算最终输出
        assert v is not None
        attn = torch.bmm(attn_probs, v)     # attn: [bsz * num_heads, tgt_len, head_dim]
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        # =========================================== Bottom@关键修改：在softmax之前注入DCA偏置 ===================================================
        
        # ONNX特殊处理
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        # 输出投影
        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)       # attn_weights: [num_heads, bsz, tgt_len, src_len]
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)     # attn_weights: [bsz, tgt_len, src_len]

        return attn, attn_weights



class DCAScalingScheduler(nn.Module):
    """
    DCA偏置缩放因子调度器 (作为nn.Module子类)
    
    支持多种调度策略：
    - constant: 固定缩放因子
    - linear_decay: 线性衰减
    - cosine_decay: 余弦衰减
    - exponential_decay: 指数衰减
    
    作为nn.Module子类，其参数会随模型一起保存和加载
    """
    
    def __init__(
        self,
        initial_scale: float = 1.0,
        final_scale: float = 0.1,
        total_steps: int = 1000,
        strategy: str = "constant"
    ):
        super().__init__()
        
        # 使用register_buffer存储可保存的参数
        self.register_buffer('initial_scale', torch.tensor(initial_scale, dtype=torch.float32))
        self.register_buffer('final_scale', torch.tensor(final_scale, dtype=torch.float32))
        self.register_buffer('total_steps', torch.tensor(total_steps, dtype=torch.long))
        self.register_buffer('current_step', torch.tensor(0, dtype=torch.long))
        
        # 策略名称存储为模块属性 (不会自动保存，需要手动处理)
        self.strategy = strategy



    def get_scale(self) -> torch.Tensor:
        """
        获取当前的缩放因子作为0维张量，以兼容torch.compile。
        """
        if self.strategy == "constant":
            return self.initial_scale
        elif self.strategy == "linear_decay":
            progress = torch.clamp(self.current_step.float() / self.total_steps.float(), 0.0, 1.0)
            scale = self.initial_scale + (self.final_scale - self.initial_scale) * progress
            return scale
        elif self.strategy == "cosine_decay":
            progress = torch.clamp(self.current_step.float() / self.total_steps.float(), 0.0, 1.0)
            cosine_factor = 0.5 * (1 + torch.cos(progress * math.pi))
            scale = self.final_scale + (self.initial_scale - self.final_scale) * cosine_factor
            return scale
        elif self.strategy == "exponential_decay":
            progress = torch.clamp(self.current_step.float() / self.total_steps.float(), 0.0, 1.0)
            exp_factor = torch.exp(-5.0 * progress)  # 衰减率为5
            scale = self.final_scale + (self.initial_scale - self.final_scale) * exp_factor
            return scale
        else:
            raise ValueError(f"Unknown scaling strategy: {self.strategy}")
            


    def step(self):
        self.current_step += 1
        

    def reset(self):
        self.current_step.zero_()

    
    def update_total_steps(self, new_total_steps: int):
        self.total_steps.fill_(new_total_steps)

    
    def get_progress(self) -> float:
        return torch.clamp(self.current_step.float() / self.total_steps.float(), 0.0, 1.0).item()

    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to include strategy"""
        state_dict = super().state_dict(destination, prefix, keep_vars)
        # 手动添加strategy到state_dict
        state_dict[prefix + 'strategy'] = self.strategy
        return state_dict
    
    
    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle strategy"""
        # 查找并提取strategy（可能有prefix）
        strategy_key = None
        for key in list(state_dict.keys()):
            if key.endswith('strategy'):
                strategy_key = key
                break
        
        if strategy_key is not None:
            self.strategy = state_dict.pop(strategy_key)
        
        # 加载其余参数
        return super().load_state_dict(state_dict, strict)

    

class ContactPredictionHead(nn.Module):
    """
    Improved ContactPredictionHead that properly handles EOS token removal using masks.
    
    This version correctly removes EOS tokens based on their actual positions in the sequence
    (using eos_mask), rather than simply slicing off the last row/column. This is crucial
    for batched sequences where EOS tokens may not be at the last position due to padding.
    """

    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        bias=True,
        eos_idx: Optional[int] = None,
        alphabet=None,  # 添加alphabet参数
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        self.alphabet = alphabet  # 存储alphabet引用
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()

        # 添加可学习的矩阵标准化器
        self.norm_attn = MatrixLinearNormalizer(
            activation='sigmoid',        # 把 logits → [0,1]
            apply_symmetrize=False,      # 前面已做
            apply_apc=False,             # 前面已做
            target_range=(0.0, 1.0),
        )



    def forward(self, tokens, attentions):
        """
        移除所有特殊token(cls/bos, eos, pad)，只保留有效残基的attention。
        支持batch内序列长度不一致的情况。

        output:
            contact_maps: 一个列表，共batch_size个元素，每个元素是形状为[valid_len_i, valid_len_i]的张量
        """
        batch_size, layers, heads, seqlen, _ = attentions.size()
        
        valid_residue_indices = []
        for b in range(batch_size):
            seq_tokens = tokens[b]  # [seqlen]
            valid_positions = []            
            for i in range(len(seq_tokens)):
                token_id = seq_tokens[i].item()
                is_special = False
                
                if self.prepend_bos and self.alphabet and hasattr(self.alphabet, 'cls_idx'):
                    if token_id == self.alphabet.cls_idx:
                        is_special = True                
                if self.append_eos and token_id == self.eos_idx:
                    is_special = True                
                if self.alphabet and hasattr(self.alphabet, 'padding_idx'):
                    if token_id == self.alphabet.padding_idx:
                        is_special = True                
                if self.alphabet and hasattr(self.alphabet, 'mask_idx'):
                    if token_id == self.alphabet.mask_idx:
                        is_special = True                
                if self.alphabet and hasattr(self.alphabet, 'unk_idx'):
                    if token_id == self.alphabet.unk_idx:
                        is_special = True                
                if not is_special:
                    valid_positions.append(i)           
            valid_residue_indices.append(torch.tensor(valid_positions, dtype=torch.long, device=tokens.device))
        
        contact_maps = []       
        for b in range(batch_size):
            valid_idx = valid_residue_indices[b]
            valid_len = len(valid_idx)
            
            if valid_len == 0:
                # 对于没有有效残基的序列，添加空接触图
                contact_maps.append(torch.zeros(0, 0, device=attentions.device, dtype=attentions.dtype))
                continue            

            # Extract valid attention for this sample: [layers, heads, valid_len, valid_len]
            valid_attention = attentions[b, :, :, valid_idx][:, :, :, valid_idx]
            
            # Reshape to combine layers and heads: [layers*heads, valid_len, valid_len]
            valid_attention = valid_attention.reshape(layers * heads, valid_len, valid_len)
            valid_attention = self.norm_attn.symmetrize_matrix(valid_attention)
            valid_attention = self.norm_attn.apply_apc_correction(valid_attention)
            valid_attention = valid_attention.to(self.regression.weight.device)
            
            # Convert processed result to required format for regression
            valid_attention = valid_attention.permute(1, 2, 0)  # [valid_len, valid_len, layers*heads]
            
            # Apply regression and activation to get contact probabilities
            contact_logits = self.regression(valid_attention).squeeze(2)  # [valid_len, valid_len]
            
            contact_prob = self.norm_attn(contact_logits, collect_penalty=self.training)
        
            contact_maps.append(contact_prob)
        
        return contact_maps



class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout

        self._init_submodules()



    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
            alphabet=self.alphabet,  # 传递alphabet参数
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )



    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        """
        关于need_head_weights和return_contacts参数的3种组合，对forward函数输出result的影响：
            * need_head_weights=False, return_contacts=False
                result = {
                    "logits": x,                   # 形状: [batch_size, seq_len, alphabet_size]
                    "representations": hidden_representations # 形状: 字典 {layer_idx: tensor[batch_size, seq_len, embed_dim]}
                }
                不包含：注意力权重、接触图

            * need_head_weights=True, return_contacts=False
                result = {
                    "logits": x,                   # 形状: [batch_size, seq_len, alphabet_size]
                    "representations": hidden_representations, # 形状: 字典 {layer_idx: tensor[batch_size, seq_len, embed_dim]}
                    "attentions": attentions       # 形状: [batch_size, num_layers, num_heads, seq_len, seq_len]
                }
                不包含：接触图

            * need_head_weights=True/False, return_contacts=True
                result = {
                    "logits": x,                   # 形状: [batch_size, seq_len, alphabet_size]
                    "representations": hidden_representations, # 形状: 字典 {layer_idx: tensor[batch_size, seq_len, embed_dim]}
                    "attentions": attentions,      # 形状: [batch_size, num_layers, num_heads, seq_len, seq_len]
                    "contacts": contacts           # 形状: [batch_size, seq_len, seq_len]
                }
                包含：完整的接触预测图

                
        output:
            results: a dict
                results["logits"]: [batch_size, seq_len, alphabet_size], 这是语言模型头的输出，表示每个位置预测下一个token的概率分布。
                results["representations"]: 字典，键为层索引，值为张量 [batch_size, seq_len, embed_dim]，含用户指定的各层表示，以及最后一层（一定包含）的表示
                    * 完整保留了所有token的'位置'，包括CLS/BOS、EOS、padding和mask token
                    * 但在padding和mask的位置，其向量值(包括嵌入层embedding向量值和中间transformer层的向量值)被置为0
                results["attentions"]：[batch_size, num_layers, num_heads, seq_len, seq_len]，保留了特殊标记（如CLS/BOS、EOS、padding）位置的权重，对于padding位置，注意力权重被掩码设为0
                results["contacts"](EvoSite修改版)：列表，包含batch_size个张量，每个形状为[valid_len_i, valid_len_i]，完全移除了特殊标记（CLS/BOS、EOS、padding），valid_len_i是有效残基数量= 原始长度 - BOS/CLS - EOS - pad
        """
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x = x.masked_fill((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            """
            关于need_head_weights和return_contacts参数的3种组合，对x, attn = layer(...)的影响：
                * 对x：x的形状始终为[seq_len, batch_size, embed_dim]，need_head_weights和return_contacts参数不会影响x的计算或形状
                * 对attn：
                    * 当need_head_weights=False时：TransformerLayer返回的attn形状为[batch_size, seq_len, seq_len]，这个attn不会被收集，因为只有当need_head_weights=True时才会将attn添加到attn_weights列表
                    * 当need_head_weights=True时：TransformerLayer返回的attn形状为[num_heads, batch_size, seq_len, seq_len]，此时在ESM2中收集attn并将其转置为[batch_size, num_heads, seq_len, seq_len]，最终将所有层的注意力权重堆叠，得到形状为[batch_size, num_layers, num_heads, seq_len, seq_len]的张量
            """
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result



    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]



class SequenceEncoder:
    """
    Utility class for encoding protein sequences to tokens using ESM alphabet.
    """
    def __init__(self, esm_alphabet):
        """
        Initialize the sequence encoder with the ESM alphabet.     
        Args:
            esm_alphabet: ESM alphabet object used for tokenization
        """
        self.alphabet = esm_alphabet
        self.batch_converter = self.alphabet.get_batch_converter()
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Encode a protein sequence to token IDs using ESM's tokenization        
        Args:
            sequence: Amino acid sequence string            
        Returns:
            Tensor of token IDs [seq_len]
        """
        _, _, tokens = self.batch_converter([("", sequence)])
        return tokens[0]
    
    def encode_batch(self, batch_sequences: List[str]) -> torch.Tensor:
        """
        Encode a batch of protein sequences to token IDs.        
        Args:
            batch_sequences: List of amino acid sequence strings            
        Returns:
            Tensor of token IDs [batch_size, seq_len]
        """
        data = [("", seq) for seq in batch_sequences]
        _, _, tokens = self.batch_converter(data)
        return tokens
    


class EvolutionaryScaleModeling(nn.Module):
    """
    Evolutionary Scale Modeling (ESM-2) with DCA feature alignment capabilities.
    This class wraps ESM-2 and adds functionality for aligning with DCA features
    through auxiliary losses or bias injection.
    """
    
    # Model mapping for ESM-2 variants with their layer counts
    ESM2_MODELS = {
        '8M': {'name': 'esm2_t6_8M_UR50D', 'layers': 6},
        '35M': {'name': 'esm2_t12_35M_UR50D', 'layers': 12},
        '150M': {'name': 'esm2_t30_150M_UR50D', 'layers': 30},
        '650M': {'name': 'esm2_t33_650M_UR50D', 'layers': 33},
        '3B': {'name': 'esm2_t36_3B_UR50D', 'layers': 36},
        '15B': {'name': 'esm2_t48_15B_UR50D', 'layers': 48},
    }
    
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        pretrained_weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        unfreeze_layers: Optional[List[int]] = None,
        use_auxiliary_losses: bool = False,
        use_concat_map: bool = True,
        use_relative_bias_injection: bool = True,
        use_single_site_potentials: bool = False,
        alignment_loss_mode: str = "kl",  # "kl", "cosine", "mse", "frobenius"
        alignment_layers: Optional[List[int]] = None,
        attention_fusion_method: str = "weighted_average",  # New parameter for attention fusion
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the ESM-2 model with DCA alignment capabilities.

        注意：
            * auxiliary_losse模式，concat maps, attention maps, coupling maps都是valid_len_i x valid_len_i的矩阵
            * bias_injection模式，DCABiasedMultiheadAttention内部attention maps, coupling maps都是seq_len x seq_len的矩阵
        
        Args:
            model_name: Name or identifier of ESM-2 model variant (e.g., "esm2_t33_650M_UR50D")
            pretrained_weights_path: Optional path to pre-trained weights
            freeze_backbone: Whether to freeze the ESM backbone by default
            unfreeze_layers: List of specific layer indices to unfreeze if backbone is frozen
                             For effective training with alignment, these should include alignment_layers
            use_auxiliary_losses: Whether to use auxiliary losses for alignment with DCA features
            use_concat_map: When use_auxiliary_losses is True, further decide whether to use concat map or attention map
            use_relative_bias_injection: Whether to use relative position bias injection
            use_single_site_potentials: Whether to use single-site potentials for Dual-head loss
            alignment_loss_mode: Type of alignment loss to use (kl, cosine, mse, frobenius)
            alignment_layers: List of layer indices to use for DCA alignment
                             Default is [middle_layer, last_layer-1, last_layer]
            attention_fusion_method: Method for fusing attention layers (weighted_average, linear_weighted, 
                                   attention_weighted, average, max, top_layer_only)
            single_site_weight: Weight for single site potential loss component
            coupling_weight: Weight for coupling matrix loss component
            device: Device to place model on
        """
        super(EvolutionaryScaleModeling, self).__init__()
        
        self.model_name = model_name
        self._device = device
        
        self.use_auxiliary_losses = use_auxiliary_losses
        self.use_concat_map = use_concat_map
        self.use_relative_bias_injection = use_relative_bias_injection
        self.use_single_site_potentials = use_single_site_potentials
        self.alignment_loss_mode = alignment_loss_mode
        self.attention_fusion_method = attention_fusion_method  # Store fusion method,一般传入weighted_average
        
        # Validate fusion method
        valid_fusion_methods = [
            "weighted_average", "linear_weighted", "attention_weighted", 
            "average", "max", "top_layer_only"
        ]
        
        if attention_fusion_method not in valid_fusion_methods:
            raise ValueError(
                f"Invalid fusion method '{attention_fusion_method}'. "
                f"Supported methods: {valid_fusion_methods}"
            )
        
        # =========================================== Top@Load ESM model and alphabet ======================================================
        logging.info(f"Loading ESM model: {model_name}")
        is_distributed = False
        rank = 0
        try:
            if dist.is_available() and dist.is_initialized():
                is_distributed = True
                rank = dist.get_rank()
                logging.info(f"Detected distributed training: rank={rank}")
        except:
            pass

        if is_distributed:
            max_retries = 3
            original_esm_model = None
            self.alphabet = None
            
            for retry_count in range(max_retries):
                try:
                    if rank != 0:
                        delay = random.uniform(0.5, 2.0) + retry_count * 0.5
                        logging.info(f"[Rank {rank}] Waiting {delay:.1f}s before loading (attempt {retry_count + 1})")
                        time.sleep(delay)
                    logging.info(f"[Rank {rank}] Loading {model_name} (attempt {retry_count + 1})")
                    original_esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
                    if original_esm_model is None or self.alphabet is None:
                        raise RuntimeError("Loaded model or alphabet is None")
                    logging.info(f"[Rank {rank}] Successfully loaded ESM model: {type(original_esm_model).__name__}")
                    break
                    
                except Exception as e:
                    logging.warning(f"[Rank {rank}] Loading attempt {retry_count + 1} failed: {e}")
                    if retry_count == max_retries - 1:
                        logging.error(f"[Rank {rank}] All {max_retries} loading attempts failed")
                        raise RuntimeError(f"Failed to load ESM model {model_name} after {max_retries} attempts: {e}")
                    else:
                        time.sleep(1.0 * (retry_count + 1))
            
            try:
                dist.barrier()
                logging.info(f"[Rank {rank}] All processes synchronized after ESM loading")
            except Exception as barrier_e:
                logging.warning(f"[Rank {rank}] Barrier sync warning: {barrier_e}")
                
        else:
            # 非分布式环境 - 直接加载
            logging.info("Loading ESM model in non-distributed mode")
            original_esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        
        # 最终验证 - 确保黑盒输出正确
        if original_esm_model is None:
            raise RuntimeError(f"ESM model loading failed: original_esm_model is None")
        if self.alphabet is None:
            raise RuntimeError(f"ESM alphabet loading failed: alphabet is None") 
        if not hasattr(original_esm_model, 'num_layers') or not hasattr(original_esm_model, 'embed_dim'):
            raise RuntimeError(f"Loaded ESM model missing required attributes")
            
        logging.info(f"   ESM model loading completed successfully:")
        logging.info(f"   Model type: {type(original_esm_model).__name__}")  
        logging.info(f"   Layers: {original_esm_model.num_layers}")
        logging.info(f"   Embed dim: {original_esm_model.embed_dim}")
        logging.info(f"   Attention heads: {original_esm_model.attention_heads}")
        logging.info(f"   Alphabet size: {len(self.alphabet)}")
        # =========================================== Bottom@Load ESM model and alphabet ======================================================
        

        # 创建自定义的ESM2实例，使用相同的配置
        self.esm_model = ESM2(
            num_layers=original_esm_model.num_layers,
            embed_dim=original_esm_model.embed_dim,
            attention_heads=original_esm_model.attention_heads,
            alphabet=self.alphabet,
            token_dropout=False             # @潜在的问题：这里token_dropout参数被设定为False，而不是默认的True
        )
        self._transfer_weights(original_esm_model, self.esm_model)
        del original_esm_model
        
        self.num_layers = self.esm_model.num_layers
        self.hidden_dim = self.esm_model.embed_dim
        
        # Create sequence encoder
        self.sequence_encoder = SequenceEncoder(self.alphabet)
        
        # Set default alignment layers if not provided
        # @潜在的问题：self.num_layers-1是否是最后一层
        # self.esm_model.layers列表的索引系统 (0-based, 标准Python列表索引)，第一层的索引是0，最后一层的索引是num_layers-1
        if alignment_layers is None:
            # Default to middle and last layers
            middle_layer = self.num_layers // 2
            alignment_layers = [middle_layer, self.num_layers-2, self.num_layers-1]
        self.alignment_layers = alignment_layers
        
        # 替换ESM模型中的MultiheadAttention为DCABiasedMultiheadAttention（如果使用偏置注入）
        if self.use_relative_bias_injection:
            self._replace_attention_modules()
        
        # =========================================== Top@放置自定义层 ====================================================== 
        # Create single-site projection head if needed
        if self.use_single_site_potentials:
            self.single_site_head = nn.Linear(self.hidden_dim, 20)

        self.norm_fused_map = MatrixLinearNormalizer(
            activation='sigmoid', 
            apply_symmetrize=True, 
            apply_apc=True, 
            target_range=(0.0, 1.0)
        )
        
        # 用于耦合矩阵辅助损失分支的规范化
        self.norm_coup_aux = MatrixLinearNormalizer(
            activation='sigmoid', 
            apply_symmetrize=True, 
            apply_apc=True, 
            target_range=(0.0, 1.0)
        )
        
        # 用于耦合矩阵偏置注入分支的规范化
        self.norm_coup_bias = MatrixLinearNormalizer(
            activation='identity',          # @潜在的问题：这里使用直接映射，没有明确将bias限制在[-1,1]
            apply_symmetrize=True, 
            apply_apc=True, 
            target_range=(-1.0, 1.0)
        )
        # =========================================== Bottom@放置自定义层 =================================================== 

        # Load custom weights if provided
        if pretrained_weights_path:
            self._load_pretrained_weights(pretrained_weights_path)
        
        # Freeze/unfreeze layers as specified
        if freeze_backbone:
            self._freeze_layers()
            if unfreeze_layers:
                self._unfreeze_layers(unfreeze_layers)
        
        # Move model to device
        self.to(device)
    


    def _load_pretrained_weights(self, path: str) -> bool:
        """
        Load pre-trained weights from the given path.
        
        Args:
            path: Path to pre-trained weights file
            
        Returns:
            success: Whether weights were successfully loaded
        """
        try:
            checkpoint = torch.load(path, map_location=self._device)
            self.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Successfully loaded weights from {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load weights from {path}: {e}")
            return False
    


    def _freeze_layers(self, layer_idx_list: Optional[List[int]] = None):
        """
        Freeze specified layers or all layers if None.
        
        Args:
            layer_idx_list: List of layer indices to freeze, or None to freeze all
        """
        # Freeze all layers first
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
    

    def _unfreeze_layers(self, layer_idx_list: List[int]):
        """
        Unfreeze specified layers and alignment layers.
        
        Args:
            layer_idx_list: List of layer indices to unfreeze
        """
        # 合并用户指定的层和对齐层
        combined_layers = list(set(layer_idx_list + self.alignment_layers))
        logging.info(f"Unfreezing user-specified layers and alignment layers: {combined_layers}")
        
        for layer_idx in combined_layers:
            if 0 <= layer_idx < self.num_layers:
                for param in self.esm_model.layers[layer_idx].parameters():
                    param.requires_grad = True
                logging.info(f"Unfrozen layer {layer_idx}")
            else:
                logging.warning(f"Layer index {layer_idx} out of range, skipping")
    


    def _transfer_weights(self, source_model, target_model):
        """
        Transfer weights from the original ESM model to our modified version.
        
        Args:
            source_model: Original ESM model loaded from pretrained
            target_model: Our modified ESM2 model
        """
        logging.info("Transferring weights from original ESM model to modified version...")
        
        try:
            # Transfer embedding weights
            logging.info("start transfer embedding layers weights...")
            try:
                target_model.embed_tokens.load_state_dict(source_model.embed_tokens.state_dict())
                logging.info("✓ Transferred embedding weights")
            except Exception as e:
                logging.error(f"❌ 传输embedding层权重失败:")
                logging.error(f"   错误位置: target_model.embed_tokens.load_state_dict()")
                logging.error(f"   错误原因: {str(e)}")
                logging.error(f"   可能原因: 源模型和目标模型的embed_tokens层结构不匹配")
                logging.error(f"   源模型embed_tokens参数: {list(source_model.embed_tokens.state_dict().keys())}")
                logging.error(f"   目标模型embed_tokens参数: {list(target_model.embed_tokens.state_dict().keys())}")
                raise RuntimeError(f"Embedding层权重传输失败: {str(e)}")
            
            # Transfer transformer layer weights
            logging.info("start transfer transformer layers weights...")
            try:
                for i, (source_layer, target_layer) in enumerate(zip(source_model.layers, target_model.layers)):
                    try:
                        target_layer.load_state_dict(source_layer.state_dict())
                        logging.debug(f"✓ Transferred layer {i} weights")
                    except Exception as e:
                        logging.error(f"❌ 传输第{i}层transformer权重失败:")
                        logging.error(f"   错误位置: target_layer.load_state_dict() at layer {i}")
                        logging.error(f"   错误原因: {str(e)}")
                        logging.error(f"   可能原因: 第{i}层的结构或参数名称不匹配")
                        logging.error(f"   源模型第{i}层参数: {list(source_layer.state_dict().keys())}")
                        logging.error(f"   目标模型第{i}层参数: {list(target_layer.state_dict().keys())}")
                        raise RuntimeError(f"第{i}层transformer权重传输失败: {str(e)}")
                
                logging.info(f"✓ Transferred {len(source_model.layers)} transformer layer weights")
            except Exception as e:
                if "transformer权重传输失败" not in str(e):
                    logging.error(f"❌ Transformer层权重传输过程中发生未预期错误:")
                    logging.error(f"   错误位置: transformer layers iteration")
                    logging.error(f"   错误原因: {str(e)}")
                    logging.error(f"   可能原因: 模型layers数量不匹配或迭代过程出错")
                    logging.error(f"   源模型layers数量: {len(source_model.layers)}")
                    logging.error(f"   目标模型layers数量: {len(target_model.layers)}")
                    raise RuntimeError(f"Transformer层权重传输失败: {str(e)}")
                else:
                    raise
            
            # Transfer layer norm weights
            logging.info("start transfer layer norm weights...")
            try:
                target_model.emb_layer_norm_after.load_state_dict(source_model.emb_layer_norm_after.state_dict())
                logging.info("✓ Transferred final layer norm weights")
            except Exception as e:
                logging.error(f"❌ 传输layer norm权重失败:")
                logging.error(f"   错误位置: target_model.emb_layer_norm_after.load_state_dict()")
                logging.error(f"   错误原因: {str(e)}")
                logging.error(f"   可能原因: 源模型和目标模型的emb_layer_norm_after层结构不匹配")
                logging.error(f"   源模型layer_norm参数: {list(source_model.emb_layer_norm_after.state_dict().keys())}")
                logging.error(f"   目标模型layer_norm参数: {list(target_model.emb_layer_norm_after.state_dict().keys())}")
                raise RuntimeError(f"Layer norm权重传输失败: {str(e)}")
            
            # Transfer LM head weights
            logging.info("start transfer LM head weights...")
            try:
                target_model.lm_head.load_state_dict(source_model.lm_head.state_dict())
                logging.info("✓ Transferred LM head weights")
            except Exception as e:
                logging.error(f"❌ 传输LM head权重失败:")
                logging.error(f"   错误位置: target_model.lm_head.load_state_dict()")
                logging.error(f"   错误原因: {str(e)}")
                logging.error(f"   可能原因: 源模型和目标模型的lm_head层结构不匹配")
                logging.error(f"   源模型lm_head参数: {list(source_model.lm_head.state_dict().keys())}")
                logging.error(f"   目标模型lm_head参数: {list(target_model.lm_head.state_dict().keys())}")
                raise RuntimeError(f"LM head权重传输失败: {str(e)}")
            
            # Transfer contact head regression weights (保持权重兼容性)
            logging.info("start transfer contact head weights...")
            try:
                target_contact_head = target_model.contact_head
                source_contact_head = source_model.contact_head
                
                # 检查contact_head是否存在
                if not hasattr(source_model, 'contact_head') or source_model.contact_head is None:
                    logging.error(f"❌ 源模型没有contact_head属性或为None")
                    raise RuntimeError("源模型缺少contact_head")
                
                if not hasattr(target_model, 'contact_head') or target_model.contact_head is None:
                    logging.error(f"❌ 目标模型没有contact_head属性或为None")
                    raise RuntimeError("目标模型缺少contact_head")
                
                # 检查regression层是否存在
                if not hasattr(source_contact_head, 'regression'):
                    logging.error(f"❌ 源模型contact_head没有regression属性")
                    logging.error(f"   源模型contact_head属性: {dir(source_contact_head)}")
                    raise RuntimeError("源模型contact_head缺少regression层")
                
                if not hasattr(target_contact_head, 'regression'):
                    logging.error(f"❌ 目标模型contact_head没有regression属性")
                    logging.error(f"   目标模型contact_head属性: {dir(target_contact_head)}")
                    raise RuntimeError("目标模型contact_head缺少regression层")
                
                # 复制regression层的权重和偏置
                try:
                    target_contact_head.regression.weight.data.copy_(source_contact_head.regression.weight.data)
                    logging.debug("✓ Copied regression weight")
                except Exception as e:
                    logging.error(f"❌ 复制regression权重失败:")
                    logging.error(f"   错误位置: target_contact_head.regression.weight.data.copy_()")
                    logging.error(f"   错误原因: {str(e)}")
                    logging.error(f"   源模型regression weight形状: {source_contact_head.regression.weight.shape}")
                    logging.error(f"   目标模型regression weight形状: {target_contact_head.regression.weight.shape}")
                    raise RuntimeError(f"Contact head regression权重复制失败: {str(e)}")
                
                try:
                    target_contact_head.regression.bias.data.copy_(source_contact_head.regression.bias.data)
                    logging.debug("✓ Copied regression bias")
                except Exception as e:
                    logging.error(f"❌ 复制regression偏置失败:")
                    logging.error(f"   错误位置: target_contact_head.regression.bias.data.copy_()")
                    logging.error(f"   错误原因: {str(e)}")
                    logging.error(f"   源模型regression bias形状: {source_contact_head.regression.bias.shape}")
                    logging.error(f"   目标模型regression bias形状: {target_contact_head.regression.bias.shape}")
                    raise RuntimeError(f"Contact head regression偏置复制失败: {str(e)}")
                
                logging.info("✓ Transferred contact head regression weights")
                
            except Exception as e:
                if "Contact head" not in str(e):
                    logging.error(f"❌ Contact head权重传输过程中发生未预期错误:")
                    logging.error(f"   错误位置: contact head weight transfer")
                    logging.error(f"   错误原因: {str(e)}")
                    raise RuntimeError(f"Contact head权重传输失败: {str(e)}")
                else:
                    raise
            
            # 验证关键参数是否匹配
            logging.info("start verify key parameters match...")
            try:
                assert target_model.contact_head.in_features == source_contact_head.in_features, \
                    f"Contact head input features mismatch: target={target_model.contact_head.in_features} vs source={source_contact_head.in_features}"
                logging.debug("✓ Input features match")
                
                assert target_model.contact_head.prepend_bos == source_contact_head.prepend_bos, \
                    f"BOS token settings mismatch: target={target_model.contact_head.prepend_bos} vs source={source_contact_head.prepend_bos}"
                logging.debug("✓ BOS token settings match")
                
                assert target_model.contact_head.append_eos == source_contact_head.append_eos, \
                    f"EOS token settings mismatch: target={target_model.contact_head.append_eos} vs source={source_contact_head.append_eos}"
                logging.debug("✓ EOS token settings match")
                
                assert target_model.contact_head.eos_idx == source_contact_head.eos_idx, \
                    f"EOS token index mismatch: target={target_model.contact_head.eos_idx} vs source={source_contact_head.eos_idx}"
                logging.debug("✓ EOS token index match")
                
            except AssertionError as e:
                logging.error(f"❌ 关键参数验证失败:")
                logging.error(f"   错误位置: 参数匹配验证")
                logging.error(f"   错误原因: {str(e)}")
                logging.error(f"   可能原因: 源模型和目标模型的contact_head配置不一致")
                raise RuntimeError(f"参数验证失败: {str(e)}")
            
            logging.info("✓ All weight transfers completed successfully with validation passed")
            
        except Exception as e:
            logging.error(f"❌ 权重传输过程中发生严重错误:")
            logging.error(f"   错误信息: {str(e)}")
            logging.error(f"   错误类型: {type(e).__name__}")
            logging.error("   建议检查源模型和目标模型的结构是否匹配")
            raise RuntimeError(f"权重传输失败: {str(e)}")
    


    def _replace_attention_modules(self):
        """
        替换ESM模型中的MultiheadAttention为DCABiasedMultiheadAttention
        
        这个方法会：
        1. 遍历所有Transformer层
        2. 将每层的self_attn替换为DCABiasedMultiheadAttention
        3. 复制原始权重到新的注意力模块
        4. 保持所有原始参数和行为不变
        """
        logging.info("Replacing MultiheadAttention modules with DCABiasedMultiheadAttention...")
        
        replaced_count = 0

        # 只替换alignment_layers
        for layer_idx in self.alignment_layers:  
            if 0 <= layer_idx < len(self.esm_model.layers):
                layer = self.esm_model.layers[layer_idx]
                old_attn = layer.self_attn      # 保存原始注意力模块
                
                # 创建新的DCA偏置注意力模块，使用相同的参数
                new_attn = DCABiasedMultiheadAttention(
                    embed_dim=old_attn.embed_dim,
                    num_heads=old_attn.num_heads,
                    kdim=old_attn.kdim,
                    vdim=old_attn.vdim,
                    dropout=old_attn.dropout,
                    bias=old_attn.out_proj.bias is not None,
                    add_bias_kv=old_attn.bias_k is not None,
                    add_zero_attn=old_attn.add_zero_attn,
                    self_attention=old_attn.self_attention,
                    encoder_decoder_attention=old_attn.encoder_decoder_attention,
                    use_rotary_embeddings=hasattr(old_attn, 'rot_emb') and old_attn.rot_emb is not None
                )
                
                # 复制所有权重和偏置
                new_attn.load_state_dict(old_attn.state_dict())
                
                # 复制特殊属性
                if hasattr(old_attn, 'prepend_bos'):
                    new_attn.prepend_bos = old_attn.prepend_bos
                else:
                    # 从alphabet获取prepend_bos设置
                    new_attn.prepend_bos = self.alphabet.prepend_bos
                    
                if hasattr(old_attn, 'append_eos'):
                    new_attn.append_eos = old_attn.append_eos
                else:
                    # 从alphabet获取append_eos设置
                    new_attn.append_eos = self.alphabet.append_eos
                
                # 替换模块
                layer.self_attn = new_attn
                del old_attn 
                replaced_count += 1
                
                logging.debug(f"Replaced attention module in layer {layer_idx}")
            else:
                raise IndexError(f"Layer index {layer_idx} is out of range (valid range: 0 to {len(self.esm_model.layers)-1})")
        
        logging.info(f"✓ Successfully replaced {replaced_count} attention modules with DCA-biased versions")


    
    def _fuse_attention_layers(
        self, 
        attention_maps: Dict[int, List[torch.Tensor]],
        fusion_method: str = "weighted_average",
    ) -> List[torch.Tensor]:
        """
        除了将attention maps的格式转换成contact_maps（即一个长度为batch_size的列表，列表的每个值是一个形状为[valid_i, valid_i]的矩阵），本函数还对attention maps进行对称化、apc、min-max

        This implementation consolidates all fusion functionality and is based on best practices from academic literature:
        - SPOT-Contact-LM (Singh et al., 2022): Layer-wise fusion improves contact prediction
        - ESM literature: Later layers capture more structural information
        - Protein contact prediction: Weighted fusion outperforms simple averaging
        
        Args:
            attention_maps: Dictionary mapping layer indices to lists of attention tensors
                          {layer_idx: [tensor1, tensor2, ..., tensorN]} where N = batch_size
            fusion_method: Method for fusing layers
                - "weighted_average": Later layers get exponentially higher weights (default, research-backed)
                - "linear_weighted": Linear weighting favoring later layers
                - "attention_weighted": Use attention intensity for weighting
                - "average": Simple average (baseline)
                - "max": Element-wise maximum
                - "top_layer_only": Use only the highest layer
            
        Returns:
            List of fused attention maps [tensor1, tensor2, ..., tensorN] where N = batch_size
            Each tensor has shape [valid_len_i, valid_len_i], valid_i can be different for different batches
        """
        if not attention_maps:
            raise ValueError("No attention maps to fuse")

        sorted_layer_indices = sorted(attention_maps.keys())
        num_layers = len(sorted_layer_indices)
        
        # Get batch size from the first layer's maps
        first_layer_maps = attention_maps[sorted_layer_indices[0]]
        batch_size = len(first_layer_maps)

        if num_layers == 1:
            return first_layer_maps # No fusion needed

        # --- Vectorization Step 1: Group samples by shape for efficient batch processing ---
        shape_groups = defaultdict(list)
        # Store (batch_idx, tensor_list) for each shape
        for b in range(batch_size):
            shape = first_layer_maps[b].shape
            # Collect all layer maps for this sample
            sample_all_layers = [attention_maps[layer_idx][b] for layer_idx in sorted_layer_indices]
            shape_groups[shape].append((b, sample_all_layers))

        # --- Vectorization Step 2: Process each shape group as a batch ---
        fused_maps_dict = {}
        for shape, group_data in shape_groups.items():
            if not shape:  # Skip empty tensors
                for b, _ in group_data:
                    fused_maps_dict[b] = torch.empty((0, 0), device=first_layer_maps[0].device, dtype=first_layer_maps[0].dtype)
                continue

            # Unzip batch indices and tensor lists
            batch_indices, sample_maps_list = zip(*group_data)
            
            # Stack all tensors for this group: [group_size, num_layers, H, W]
            group_tensor = torch.stack([torch.stack(s_maps) for s_maps in sample_maps_list])
            group_size, _, H, W = group_tensor.shape
            
            fused_group = torch.zeros(group_size, H, W, device=group_tensor.device, dtype=group_tensor.dtype)

            # --- Vectorization Step 3: Apply fusion logic to the entire group tensor ---
            if fusion_method == "top_layer_only":
                fused_group = group_tensor[:, -1, :, :]
            else:
                weights = torch.ones(num_layers, device=group_tensor.device, dtype=group_tensor.dtype) # Default for 'average'
                if fusion_method in ["weighted_average", "linear_weighted", "attention_weighted"]:
                    if fusion_method == "weighted_average":
                        layer_positions = torch.arange(num_layers, device=group_tensor.device, dtype=group_tensor.dtype)
                        weights = torch.exp(layer_positions * 0.5)
                    elif fusion_method == "linear_weighted":
                        weights = torch.arange(1, num_layers + 1, device=group_tensor.device, dtype=group_tensor.dtype)
                    elif fusion_method == "attention_weighted":
                        # Compute weights based on intensity: [group_size, num_layers]
                        intensities = group_tensor.mean(dim=(-1, -2))
                        weights = F.softmax(intensities, dim=-1) # Shape becomes [group_size, num_layers]

                if weights.dim() == 1: # For non-adaptive weights
                    weights = weights / weights.sum()
                    # Reshape for broadcasting: [1, num_layers, 1, 1]
                    weights = weights.view(1, -1, 1, 1) 
                else: # For attention_weighted
                    # Reshape for broadcasting: [group_size, num_layers, 1, 1]
                    weights = weights.unsqueeze(-1).unsqueeze(-1)
                
                # Perform weighted sum
                fused_group = (group_tensor * weights).sum(dim=1)

            # Store fused maps for this group
            for i, b_idx in enumerate(batch_indices):
                fused_maps_dict[b_idx] = fused_group[i]
        
        # --- Final Step: Reconstruct batch in original order and normalize ---
        final_fused_maps = []
        for b in range(batch_size):
            fused_map = fused_maps_dict[b]
            # Normalize the fused map
            normalized_map = self.norm_fused_map(fused_map, collect_penalty=self.training)
            final_fused_maps.append(normalized_map)
            
        return final_fused_maps
    


    def kl_divergence_loss(
        self,
        prediction_maps: List[torch.Tensor],
        coupling_matrices: List[torch.Tensor],
        eps: float = 1e-8,
        max_logit_value: float = 10.0
    ) -> torch.Tensor:
        """
        Vectorized and numerically stable KL divergence loss, compatible with the original training logic.
        This function treats prediction_maps as logits and applies softmax, ensuring backward compatibility
        with models trained using the logic from kl_divergence_loss_stable.

        Args:
            prediction_maps: List of tensors [valid_len_i, valid_len_i], treated as logits.
            coupling_matrices: List of DCA coupling matrices [valid_len_i, valid_len_i], treated as raw scores.
            eps: Small constant for numerical stability.
            max_logit_value: Value to clamp logits to, preventing overflow in softmax.

        Returns:
            Average KL divergence loss as a scalar tensor.
        """
        if not prediction_maps or not any(p.numel() > 0 for p in prediction_maps):
            # Handle cases with no valid data to prevent errors with max() on empty sequence
            return torch.tensor(0.0, device=self._device if hasattr(self, 'device') else 'cpu', requires_grad=True)

        # Filter out empty tensors which can occur in a batch
        valid_indices = [i for i, p in enumerate(prediction_maps) if p.numel() > 0]
        if not valid_indices:
            return torch.tensor(0.0, device=prediction_maps[0].device, requires_grad=True)

        prediction_maps = [prediction_maps[i] for i in valid_indices]
        coupling_matrices = [coupling_matrices[i] for i in valid_indices]

        device = prediction_maps[0].device
        dtype = prediction_maps[0].dtype

        # --- Vectorization Step 1: Pad lists of tensors to create batch tensors ---
        pred_flat = [p.flatten() for p in prediction_maps]
        coupl_flat = [c.flatten() for c in coupling_matrices]
        
        padded_preds_flat = pad_sequence(pred_flat, batch_first=True, padding_value=0)
        padded_coupls_flat = pad_sequence(coupl_flat, batch_first=True, padding_value=0)

        # --- Vectorization Step 2: Create masks and reshape back to matrices ---
        lengths = [p.shape[0] for p in prediction_maps]
        max_len = max(lengths)
        
        padded_preds = padded_preds_flat.view(-1, max_len, max_len)
        padded_coupls = padded_coupls_flat.view(-1, max_len, max_len)

        batch_size = len(lengths)
        arange = torch.arange(max_len, device=device)
        seq_mask = arange[None, :] < torch.tensor(lengths, device=device)[:, None]
        pair_mask = seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)
        diag_mask = torch.eye(max_len, dtype=torch.bool, device=device).unsqueeze(0)
        final_mask = pair_mask & ~diag_mask

        # --- Step 3: Replicate the exact mathematical operations of the original stable function ---

        # 3.1) Treat prediction_maps as logits, clamp, and apply softmax.
        # Clamp logits to prevent overflow/underflow in softmax, mirroring `max_value`.
        padded_preds = torch.clamp(padded_preds, min=-max_logit_value, max=max_logit_value)
        # Mask out invalid positions with a large negative number before softmax.
        padded_preds_masked = padded_preds.masked_fill(~final_mask, -1e9)
        # Apply softmax row-wise. This is the critical step for compatibility.
        pred_prob = F.softmax(padded_preds_masked, dim=-1)

        # 3.2) Treat coupling_matrices as raw scores and normalize by sum.
        # This mirrors the logic of the original `_fallback_kl_computation` which `kl_divergence_loss_stable`
        # assumes is done for the target.
        row_sum_coupls = (padded_coupls * final_mask).sum(dim=-1, keepdim=True)
        coupl_prob = padded_coupls / (row_sum_coupls + eps)

        # 3.3) Clamp probabilities for numerical stability.
        pred_prob = torch.clamp(pred_prob, min=eps)
        coupl_prob = torch.clamp(coupl_prob, min=eps)
        
        # Re-normalize after clamp to ensure they are valid distributions
        pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True)
        coupl_prob = coupl_prob / coupl_prob.sum(dim=-1, keepdim=True)

        # --- Step 4: Compute KL Divergence using a stable method ---
        # Using PyTorch's built-in kl_div is numerically stable and recommended.
        # It computes target * (log(target) - log(input)).
        # We provide log(pred_prob) as input.
        log_pred_prob = torch.log(pred_prob)
        kl_div_elements = F.kl_div(log_pred_prob, coupl_prob, reduction='none')

        # Mask out invalid elements (padding, diagonal) from the final loss value.
        kl_div_elements.masked_fill_(~final_mask, 0)

        # --- Step 5: Final Loss Aggregation ---
        # Sum the KL divergence for each row, then take the mean over all valid rows in the batch.
        # This correctly averages the per-sample loss, matching the logic of the original loop.
        row_kl_sum = kl_div_elements.sum(dim=-1) # [batch_size, max_len]
        total_kl_sum = row_kl_sum.sum() # Sum of KL divergences of all valid rows
        num_valid_rows = seq_mask.sum() # Total number of valid rows across the batch
        
        # Final average loss
        total_loss = total_kl_sum / num_valid_rows.clamp(min=1)

        # Final check for NaN/Inf as a safeguard.
        if not torch.isfinite(total_loss):
            logging.warning(f"Non-finite KL divergence detected: {total_loss.item()}. Returning zero loss.")
            return torch.zeros(1, device=device, dtype=dtype, requires_grad=True)

        return total_loss



    def cosine_similarity_loss(
        self,
        prediction_maps: List[torch.Tensor],
        coupling_matrices: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss between prediction maps and coupling matrices.
        
        Args:
            prediction_maps: List of tensors [valid_len_i, valid_len_i] 
                        (either attention maps or contact maps)
            coupling_matrices: List of DCA coupling matrices [valid_len_i, valid_len_i]
            
        Returns:
            Average cosine similarity loss (1 - cosine similarity)
        """
        if len(prediction_maps) != len(coupling_matrices):
            raise ValueError(f"Number of prediction maps ({len(prediction_maps)}) "
                            f"must match number of coupling matrices ({len(coupling_matrices)})")
        
        if not prediction_maps:
            raise ValueError("Empty prediction maps provided")
        
        total_loss = 0.0
        batch_size = len(prediction_maps)
        
        for i in range(batch_size):
            pred_map = prediction_maps[i]  # [valid_len_i, valid_len_i]
            coupl_matrix = coupling_matrices[i]  # [valid_len_i, valid_len_i]
            
            # Check for compatible shapes
            if pred_map.shape != coupl_matrix.shape:
                raise ValueError(f"Shape mismatch at index {i}: prediction map {pred_map.shape} vs "
                            f"coupling matrix {coupl_matrix.shape}")
            
            # Flatten for cosine similarity computation
            pred_flat = pred_map.flatten()
            coupl_flat = coupl_matrix.flatten()
            
            # Check for zero vectors
            pred_norm = torch.norm(pred_flat)
            coupl_norm = torch.norm(coupl_flat)
            
            if pred_norm == 0 or coupl_norm == 0:
                logging.warning(f"Zero vector detected in sample {i}, adding small epsilon")
                # Add small epsilon to avoid division by zero
                pred_flat = pred_flat + 1e-8
                coupl_flat = coupl_flat + 1e-8
            
            # Compute cosine similarity
            cos_sim = F.cosine_similarity(pred_flat.unsqueeze(0), coupl_flat.unsqueeze(0))
            
            # Loss is 1 - cosine similarity
            total_loss = total_loss + (1.0 - cos_sim)
        
        # Average loss across batch
        return total_loss / batch_size



    def mse_loss(
        self,
        prediction_maps: List[torch.Tensor],
        coupling_matrices: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute MSE loss between prediction maps and coupling matrices.
        
        Args:
            prediction_maps: List of tensors [valid_len_i, valid_len_i] 
                        (either attention maps or contact maps)
            coupling_matrices: List of DCA coupling matrices [valid_len_i, valid_len_i]
            
        Returns:
            Average MSE loss
        """
        if len(prediction_maps) != len(coupling_matrices):
            raise ValueError(f"Number of prediction maps ({len(prediction_maps)}) "
                            f"must match number of coupling matrices ({len(coupling_matrices)})")
        
        if not prediction_maps:
            raise ValueError("Empty prediction maps provided")
        
        total_loss = 0.0
        batch_size = len(prediction_maps)
        
        for i in range(batch_size):
            pred_map = prediction_maps[i]  # [valid_len_i, valid_len_i]
            coupl_matrix = coupling_matrices[i]  # [valid_len_i, valid_len_i]
            
            # Check for compatible shapes
            if pred_map.shape != coupl_matrix.shape:
                raise ValueError(f"Shape mismatch at index {i}: prediction map {pred_map.shape} vs "
                            f"coupling matrix {coupl_matrix.shape}")
            
            # Compute MSE loss
            mse = F.mse_loss(pred_map, coupl_matrix)
            total_loss = total_loss + mse
        
        # Average loss across batch
        return total_loss / batch_size



    def frobenius_norm_loss(
        self,
        prediction_maps: List[torch.Tensor],
        coupling_matrices: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Frobenius norm loss between prediction maps and coupling matrices.
        
        Args:
            prediction_maps: List of tensors [valid_len_i, valid_len_i] 
                        (either attention maps or contact maps)
            coupling_matrices: List of DCA coupling matrices [valid_len_i, valid_len_i]
            
        Returns:
            Average Frobenius norm loss
        """
        if len(prediction_maps) != len(coupling_matrices):
            raise ValueError(f"Number of prediction maps ({len(prediction_maps)}) "
                            f"must match number of coupling matrices ({len(coupling_matrices)})")
        
        if not prediction_maps:
            raise ValueError("Empty prediction maps provided")
        
        total_loss = 0.0
        batch_size = len(prediction_maps)
        
        for i in range(batch_size):
            pred_map = prediction_maps[i]  # [valid_len_i, valid_len_i]
            coupl_matrix = coupling_matrices[i]  # [valid_len_i, valid_len_i]
            
            # Check for compatible shapes
            if pred_map.shape != coupl_matrix.shape:
                raise ValueError(f"Shape mismatch at index {i}: prediction map {pred_map.shape} vs "
                            f"coupling matrix {coupl_matrix.shape}")
            
            # Compute Frobenius norm
            diff = pred_map - coupl_matrix
            frob_norm = torch.norm(diff, p='fro')
            total_loss = total_loss + frob_norm
        
        # Average loss across batch
        return total_loss / batch_size



    def dual_head_loss_for_single_site(
        self,
        logits: torch.Tensor,
        dca_single_site_potentials: List[torch.Tensor],
        tokens: Optional[torch.Tensor] = None,
        valid_residue_indices: Optional[List[torch.Tensor]] = None,
        temperature: float = 1.0,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute KL divergence loss between predicted amino acid distributions and 
        DCA single-site potentials, handling variable sequence lengths.
        
        Args:
            logits: Predicted amino acid logits [batch_size, seq_len, 20]
                    Contains all positions including special tokens
            dca_single_site_potentials: List of tensors, each [valid_len_i, 20]
                    Each tensor contains only valid residue positions
            tokens: Input token IDs [batch_size, seq_len] for identifying valid positions
            valid_residue_indices: List of valid residue indices for each sequence  
            temperature: Temperature for softmax
            reduction: Reduction method ("none", "mean", "sum")
            
        Returns:
            KL divergence loss
        """
        # =========================================== Top@check ====================================================== 
        # Enhanced input validation
        if logits is None:
            raise ValueError("Logits tensor cannot be None")
        if not dca_single_site_potentials:
            raise ValueError("DCA single-site potentials cannot be None or empty")
        
        # Check batch size consistency
        batch_size = logits.size(0)
        if len(dca_single_site_potentials) != batch_size:
            raise ValueError(f"Number of dca_single_site_potentials tensors ({len(dca_single_site_potentials)}) "
                            f"must match batch size ({batch_size})")
        
        # Validate parameters
        if temperature <= 0:
            print(f"Warning: Invalid temperature {temperature}, using 1.0")
            temperature = 1.0
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        # =========================================== Bottom@check =================================================== 


        # collect losses for each sample
        sample_losses = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Get potentials for this sample (contains only valid positions)
            sample_potentials = dca_single_site_potentials[i]  # [valid_len_i, 20]
            valid_len = sample_potentials.size(0)
            
            if valid_residue_indices is not None and i < len(valid_residue_indices):
                valid_idx = valid_residue_indices[i].to(logits.device)
                valid_logits = logits[i, valid_idx, :]  # [valid_len_i, 20]
                
                # Check
                if valid_logits.size(0) != valid_len:
                    raise ValueError(f"Valid logits length ({valid_logits.size(0)}) doesn't match "
                                f"potentials length ({valid_len}) for sample {i}. "
                                f"Expected {valid_len} but got {valid_logits.size(0)}.")
            elif tokens is not None:
                # Fall back to calculating from tokens if valid_residue_indices not provided
                # Identify valid positions (non-special tokens)
                valid_positions = []
                for j, token_id in enumerate(tokens[i]):
                    is_special = False
                    
                    # Check for special tokens
                    if token_id == self.alphabet.padding_idx:
                        is_special = True
                    if self.alphabet.prepend_bos and token_id == self.alphabet.cls_idx:
                        is_special = True
                    if self.alphabet.append_eos and token_id == self.alphabet.eos_idx:
                        is_special = True
                    if hasattr(self.alphabet, 'mask_idx') and token_id == self.alphabet.mask_idx:
                        is_special = True
                    if hasattr(self.alphabet, 'unk_idx') and token_id == self.alphabet.unk_idx:
                        is_special = True
                    
                    if not is_special:
                        valid_positions.append(j)
                
                # Extract valid positions from logits
                valid_idx = torch.tensor(valid_positions, device=logits.device)
                valid_logits = logits[i, valid_idx, :]  # [valid_len_i, 20]
                
                # Check lengths match between valid logits and potentials
                if valid_logits.size(0) != valid_len:
                    raise ValueError(f"Valid logits length ({valid_logits.size(0)}) doesn't match "
                                f"potentials length ({valid_len}) for sample {i}. "
                                f"Expected {valid_len} but got {valid_logits.size(0)}.")
            else:
                logging.warning("No tokens tensor provided, using heuristic approach to extract valid positions.")
                start_idx = 1 if self.alphabet.prepend_bos else 0
                end_idx = min(start_idx + valid_len, logits.size(1))
                valid_logits = logits[i, start_idx:end_idx, :]
                if valid_logits.size(0) < valid_len:
                    raise ValueError(f"Could not extract {valid_len} valid positions from logits. "
                                f"Please provide tokens tensor for accurate position mapping.")
                valid_logits = valid_logits[:valid_len]
            

            # Apply temperature scaling
            scaled_logits = valid_logits / temperature
            
            # Compute softmax distributions
            pred_probs = F.softmax(scaled_logits, dim=-1)
            target_probs = F.softmax(sample_potentials, dim=-1)
            
            # Compute KL divergence
            kl_div = F.kl_div(
                pred_probs.log(), 
                target_probs,
                reduction='none'
            ).sum(dim=-1)  # [valid_len_i]
            
            # Compute mean loss for this sample
            sample_losses.append(kl_div.mean())
        
        # Stack losses and apply reduction
        if sample_losses:
            batch_loss = torch.stack(sample_losses)
            
            if reduction == "none":
                return batch_loss
            elif reduction == "sum":
                return batch_loss.sum()
            else:  # mean
                return batch_loss.mean()
        else:
            # Return zero loss if no samples were processed
            return torch.tensor(0.0, device=logits.device)
    


    def compute_coupling_alignment_loss(
        self,
        contact_maps: Optional[List[torch.Tensor]] = None,
        attention_maps: Optional[List[torch.Tensor]] = None,
        dca_coupling_matrix: List[torch.Tensor] = None,
        mode: str = "kl"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute alignment loss between ESM maps and DCA coupling matrix.
        
        Args:
            contact_maps: List of tensors, each with shape [valid_len_i, valid_len_i]
            attention_maps: List of tensors, each with shape [valid_len_i, valid_len_i]
            dca_coupling_matrix: List of tensors, each with shape [valid_len_i, valid_len_i]
            mode: Loss mode (kl, cosine, mse, frobenius)
            
        Returns:
            Dictionary of computed losses
        """
        losses = {}
        
        # Choose which maps to use (contact_maps or attention_maps)
        prediction_maps = contact_maps if contact_maps is not None else attention_maps
        
        if prediction_maps is None:
            raise ValueError("Either contact_maps or attention_maps must be provided")
        
        if dca_coupling_matrix is None:
            raise ValueError("DCA coupling matrix cannot be None")
        
        # 对耦合矩阵进行归一化处理到[0,1]范围
        normalized_coupling_matrices = []
        for matrix in dca_coupling_matrix:
            normalized_matrix = self.norm_coup_aux(matrix, collect_penalty=self.training)
            normalized_coupling_matrices.append(normalized_matrix)
            
        # Choose loss function based on mode
        if mode == "kl":
            coupling_loss = self.kl_divergence_loss(prediction_maps, normalized_coupling_matrices)
        elif mode == "cosine":
            coupling_loss = self.cosine_similarity_loss(prediction_maps, normalized_coupling_matrices)
        elif mode == "mse":
            coupling_loss = self.mse_loss(prediction_maps, normalized_coupling_matrices)
        elif mode == "frobenius":
            coupling_loss = self.frobenius_norm_loss(prediction_maps, normalized_coupling_matrices)
        else:
            raise ValueError(f"Unsupported alignment loss mode: {mode}")
        
        losses['coupling_loss'] = coupling_loss
        
        return losses



    def compute_range_penalties(self) -> Dict[str, torch.Tensor]:
        """
        Compute range penalties from active normalizers based on model configuration.
        
        Returns:
            Dictionary containing individual penalties and total penalty
        """
        penalties = {}
        device = next(self.parameters()).device
        
        # 只有在辅助损失模式且使用注意力图(而非接触图)时才收集norm_fused_map的惩罚
        if self.use_auxiliary_losses and not self.use_concat_map:
            if hasattr(self, 'norm_fused_map'):
                penalties['fused_map_penalty'] = self.norm_fused_map.range_penalty()
        
        # 只有在辅助损失模式且使用接触图(而非注意力图)时才收集norm_attn的惩罚
        if self.use_auxiliary_losses and self.use_concat_map:
            penalties['contact_head_penalty'] = self.esm_model.contact_head.norm_attn.range_penalty()
        
        # 使用辅助损失进行对齐下收集norm_coup_aux的惩罚:
        if self.use_auxiliary_losses:
            if hasattr(self, 'norm_coup_aux'):
                penalties['coupling_aux_penalty'] = self.norm_coup_aux.range_penalty()

        # norm_coup_bias不需要惩罚，它是用于bias injection的，且使用了identity激活
        
        # 计算总惩罚
        if penalties:
            total_penalty = sum(penalties.values())
            penalties['total_penalty'] = total_penalty
        else:
            # 如果没有收集到任何惩罚，返回零张量作为总惩罚
            penalties['total_penalty'] = torch.tensor(0.0, device=device)
        
        return penalties
    


    def remove_dca_bias(self, preserve_scaling: bool = False):
        """移除所有DCA偏置"""
        if self.use_relative_bias_injection:
            # 对于DCABiasedMultiheadAttention，直接清除偏置
            removed_count = 0
            for layer_idx in self.alignment_layers:  # 只清除alignment_layers中的层
                if 0 <= layer_idx < len(self.esm_model.layers):
                    layer = self.esm_model.layers[layer_idx]
                    if isinstance(layer.self_attn, DCABiasedMultiheadAttention):
                        layer.self_attn.clear_dca_bias(preserve_scaling=preserve_scaling)
                        removed_count += 1
            
            # if removed_count > 0:
            #     logging.info(f"Removed DCA bias from {removed_count} attention layers")



    def inject_dca_bias(self, dca_coupling_matrices, scaling_factor=1.0, layer_indices=None):
        """
        DCA偏置注入，支持为批次中的每个样本设置不同的DCA矩阵

        本函数以及DCABiasedMultiheadAttention.prepare_dca_bias都是处理数据集中样本的DCA耦合矩阵，
            * 本函数对DCA耦合矩阵进行归一化
            * DCABiasedMultiheadAttention.prepare_dca_bias函数对耦合矩阵填充至seq_len，并且乘以scaling_factor
        
        Args:
            dca_coupling_matrices: DCA耦合矩阵列表，每个[valid_len_i, valid_len_i]
            scaling_factor: 控制耦合偏置的强度，默认为1.0
            layer_indices: 要注入的层索引，默认使用self.alignment_layers
        """
        if not isinstance(dca_coupling_matrices, list):
            raise TypeError("Expected dca_coupling_matrices to be a list of tensors")
        
        if not self.use_relative_bias_injection:
            raise ValueError("Relative bias injection is disabled. DCA bias will not be applied.")

        target_layers = layer_indices if layer_indices is not None else self.alignment_layers
        
        # 归一化耦合矩阵到[-1,1]范围
        normalized_matrices = []
        for matrix in dca_coupling_matrices:
            normalized_matrix = self.norm_coup_bias(matrix, collect_penalty=False)      
            normalized_matrices.append(normalized_matrix)
        
        # 为每个目标层设置偏置列表
        applied_count = 0
        for layer_idx in target_layers:
            if 0 <= layer_idx < len(self.esm_model.layers):
                layer = self.esm_model.layers[layer_idx]
                
                if isinstance(layer.self_attn, DCABiasedMultiheadAttention):
                    layer.self_attn.set_dca_bias(
                        bias_matrices=normalized_matrices,
                        scaling_factor=scaling_factor
                    )
                    applied_count += 1
                    # logging.debug(f"Applied batch DCA bias to layer {layer_idx}")
            else:
                logging.warning(f"Layer {layer_idx} does not have DCABiasedMultiheadAttention.")
        
        # logging.info(f"Applied batch DCA coupling bias to {applied_count} attention layers")


    
    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        sequences: Optional[List[str]] = None,
        dca_coupling_matrix: Optional[torch.Tensor] = None,
        single_site_potentials: Optional[torch.Tensor] = None,
        repr_layers: Optional[List[int]] = None,
        dynamic_scaling_factor: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through the ESM-2 model with optional DCA alignment.

        forward方法的输出是一个包含以下可能键的字典：
            results = {
                # 基础输出 (始终包含)
                'embeddings': ...,               # 残基嵌入 [batch_size, seq_len, hidden_dim]，embeddings是最后一层Transformer的输出，包含序列中每个位置的向量表示，包括特殊标记（如CLS/BOS、EOS、padding）。
                'valid_tokens_mask': ...,        # 有效token掩码 [batch_size, seq_len]，1表示有效残基（非特殊token），0表示特殊
                'valid_residue_indices': ...,    # 有效残基索引列表，长度为batch_size的列表，每个元素是[valid_len_i]的一维tensor，表示每个样本有效残基的位置索引。
                
                # 可选输出 (取决于配置和输入)
                'representations': ...,          # 层表示字典 {layer: tensor}，用户指定层的输出（如第32层、第33层等），包含所有token位置。仅当repr_layers不为空时
                'contact_maps': ...,             # 接触图列表，长度为batch_size的列表，每个元素是形状为[valid_len_i, valid_len_i]的张量，完全移除了特殊标记，仅当use_concat_map=True时
                'attention_maps': ...,           # 注意力图列表，长度为batch_size的列表，每个元素是形状为[valid_len_i, valid_len_i]的张量，完全移除了特殊标记，仅当use_concat_map=False时
                'single_site_logits': ...,       # 单点位氨基酸分布logits，[batch_size, seq_len, 20],每个残基的20种氨基酸分布logits，包含所有token位置。仅当use_single_site_potentials=True时
                'single_site_loss': ...,         # 单点位KL散度损失，一个float tensor,仅当use_single_site_potentials=True时
                'coupling_losses': ...,          # 耦合对齐损失，字典，仅当self.use_auxiliary_losses==True
                'range_penalties': ...,          # 范围惩罚字典，仅当self.training==True时
                'range_penalty_loss': ...,       # 总范围惩罚损失，一个float tensor,仅当self.training==True时
            }

            * 'coupling_losses': {                # 耦合对齐损失
                    'coupling_loss': tensor         # 标量，耦合损失
                },
            * 'range_penalties': {                # 范围惩罚字典
                    'fused_map_penalty': tensor,    # 条件性
                    'contact_head_penalty': tensor, # 条件性
                    'coupling_aux_penalty': tensor, # 条件性
                    'total_penalty': tensor         # 总惩罚
                },
            * 'range_penalty_loss': tensor        # 总范围惩罚损失


        Args:
            tokens: Input token IDs [batch_size, seq_len]. If None, sequences must be provided.
            sequences: List of sequence strings. Used if tokens is None.
            dca_coupling_matrix: 2维张量列表：[coupling_matrix_0, coupling_matrix_1,coupling_matrix_2,...]，其中coupling_matrix_i形状为：[valid_len_i, valid_len_i]
            single_site_potentials: 2维张量列表：[single_site_potential_0, single_site_potential_1,single_site_potential_2,...]，其中single_site_potential_i形状为：[valid_len_i, 20]
            repr_layers: Which layers to extract representations from. 注意repr_layers的索引系统是1-based for transformer layers，即第一个transformer层索引是1，最后一个transformer层的索引是num_layers，所以如果想要访问第32层的hidden state，需指定repr_layers=[32]
            
        Returns:
            Dict
        """
        # Handle either tokens or sequences as input
        if tokens is None and sequences is not None:
            tokens = self.sequence_encoder.encode_batch(sequences).to(self._device)
        elif tokens is None and sequences is None:
            raise ValueError("Either tokens or sequences must be provided")
        tokens = tokens.to(self._device)
        batch_size, seq_len = tokens.shape


        # =========================================== Top@Create valid token masks ====================================================== 
        # 1. Create binary mask for valid tokens (1 = valid, 0 = special token)
        valid_tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
        
        # Mask special tokens
        valid_tokens_mask &= (tokens != self.alphabet.padding_idx)
        if self.alphabet.prepend_bos:
            valid_tokens_mask &= (tokens != self.alphabet.cls_idx)
        if self.alphabet.append_eos:
            valid_tokens_mask &= (tokens != self.alphabet.eos_idx)
        if hasattr(self.alphabet, 'mask_idx'):
            valid_tokens_mask &= (tokens != self.alphabet.mask_idx)
        if hasattr(self.alphabet, 'unk_idx'):
            valid_tokens_mask &= (tokens != self.alphabet.unk_idx)
        
        # 2. Create list of valid position indices for each sequence
        valid_residue_indices = []
        for b in range(batch_size):
            valid_positions = torch.nonzero(valid_tokens_mask[b], as_tuple=True)[0]
            valid_residue_indices.append(valid_positions)
        # =========================================== Bottom@Create valid token masks =================================================== 



        # =========================== Top@set up layer representation extraction ==================================== #
        # 对于self.esm_model.forward 的参数 repr_layers，索引系统是1-based for transformer layers，第一个transformer层索引是1，最后一个transformer层的索引是num_layers
        if repr_layers is None:
            repr_layers = []

        requested_layers = list(set(repr_layers))
        if self.num_layers not in requested_layers:
            requested_layers.append(self.num_layers)
        # =========================== Bottom@set up layer representation extraction ==================================== #

        
        results = {}

        results['valid_tokens_mask'] = valid_tokens_mask
        results['valid_residue_indices'] = valid_residue_indices

        # =========================== Top@auxiliary losses ============================================================================== #
        if dca_coupling_matrix is not None and self.use_auxiliary_losses:
            

            # Determine if we need attention weights for auxiliary losses
            need_attention_map = (dca_coupling_matrix is not None and self.use_auxiliary_losses and not self.use_concat_map)
            need_concat_map = (dca_coupling_matrix is not None and self.use_auxiliary_losses and self.use_concat_map)
            
            # Use ESM's standard forward pass
            output = self.esm_model(
                tokens,
                repr_layers=requested_layers,
                return_contacts=need_concat_map,        
                need_head_weights=need_attention_map  
            )

            results['embeddings'] = output['representations'][self.num_layers]

            if repr_layers:
                results['representations'] = {
                    layer: output['representations'][layer] 
                    for layer in repr_layers 
                    if layer in output['representations']
                }



            # Compute single-site logits if needed
            if self.use_single_site_potentials:
                single_site_logits = self.single_site_head(results['embeddings'])
                results['single_site_logits'] = single_site_logits

            # Compute single site potential alignment loss (always use dual head loss)
            if (self.use_single_site_potentials and single_site_potentials is not None and 'single_site_logits' in results):
                single_site_loss = self.dual_head_loss_for_single_site(
                    logits=results['single_site_logits'],
                    dca_single_site_potentials=single_site_potentials,
                    tokens=tokens,
                    valid_residue_indices=valid_residue_indices,  # 传递已计算的valid_residue_indices
                    temperature=1.0,
                    reduction="mean"
                )
                results['single_site_loss'] = single_site_loss



            if self.use_concat_map and 'contacts' in output and output['contacts'] is not None:
                # print('Use contacts map for coupling alignment')
                results['contact_maps'] = output['contacts']
            elif not self.use_concat_map and 'attentions' in output and output['attentions'] is not None:
                # ESM attention shape: [batch_size, num_layers, num_heads, seq_len, seq_len]
                esm_attentions = output['attentions']
                
                # 1. Vectorized selection of alignment layers
                layer_indices_tensor = torch.tensor(self.alignment_layers, device=esm_attentions.device)    # 创建一个包含需要对齐的层索引（例如 [16, 31, 32]）的张量。
                selected_attentions = esm_attentions.index_select(1, layer_indices_tensor) # [B, num_alignment_layers, H, T, T]

                # 2. Vectorized averaging across attention heads
                avg_head_attentions = selected_attentions.mean(dim=2) # [B, num_alignment_layers, T, T]

                # --- EFFICIENT DATA EXTRACTION (Loop is unavoidable for variable lengths but computation is minimized) ---
                attention_maps_by_layer = {}
                for i, layer_idx in enumerate(self.alignment_layers):
                    layer_maps = []
                    for b in range(batch_size):
                        valid_idx = valid_residue_indices[b]
                        if valid_idx.numel() > 0:
                            # Use advanced indexing for efficient slicing (no heavy computation here)
                            valid_map = avg_head_attentions[b, i, valid_idx[:, None], valid_idx]
                            layer_maps.append(valid_map)
                        else:
                            # Handle empty sequences
                            layer_maps.append(torch.empty(0, 0, device=tokens.device, dtype=esm_attentions.dtype))
                    attention_maps_by_layer[layer_idx] = layer_maps
                
                # Fuse the collected maps (the fusion function is also vectorized)
                results['attention_maps'] = self._fuse_attention_layers(
                    attention_maps_by_layer, 
                    fusion_method=self.attention_fusion_method
                )
            else:
                raise RuntimeError("Could not obtain maps for auxiliary loss. Check model output.")
                
                
            # Compute auxiliary losses for coupling matrix alignment
            if self.use_concat_map and 'contact_maps' in results and results['contact_maps'] is not None:
                coupling_losses = self.compute_coupling_alignment_loss(
                    contact_maps=results['contact_maps'],
                    dca_coupling_matrix=dca_coupling_matrix,
                    mode=self.alignment_loss_mode
                )
            elif not self.use_concat_map and 'attention_maps' in results and results['attention_maps'] is not None:
                coupling_losses = self.compute_coupling_alignment_loss(
                    attention_maps=results['attention_maps'],
                    dca_coupling_matrix=dca_coupling_matrix,
                    mode=self.alignment_loss_mode
                )
            else:
                raise RuntimeError("error occur in auxiliary losses module, type 2.")
            
            results['coupling_losses'] = coupling_losses
        # =========================== Bottom@auxiliary losses ======================================================================= #



        # =========================== Top@bias injection ============================================================================= #
        # Handle coupling matrix bias injection
        elif dca_coupling_matrix is not None and self.use_relative_bias_injection:
            # Ensure dca_coupling_matrix is a list as expected by inject_dca_bias
            coupling_matrices = dca_coupling_matrix if isinstance(dca_coupling_matrix, list) else [dca_coupling_matrix]
            
            try:
                # 首先清除任何现有偏置
                self.remove_dca_bias()

                # Use dynamic scaling factor if provided, otherwise use default
                scaling_factor = dynamic_scaling_factor if dynamic_scaling_factor is not None else 1.0

                # Inject DCA coupling matrix as bias into attention layers 
                self.inject_dca_bias(
                    dca_coupling_matrices=coupling_matrices, 
                    scaling_factor=scaling_factor
                )
                
                # Perform forward pass with bias injection active
                output = self.esm_model(
                    tokens,
                    repr_layers=requested_layers,
                    need_head_weights=True
                )
                
                # Extract results
                results['embeddings'] = output['representations'][self.num_layers]
                
                if repr_layers:
                    results['representations'] = {
                        layer: output['representations'][layer] 
                        for layer in repr_layers 
                        if layer in output['representations']
                    }
                

                # Compute single-site logits if needed
                if self.use_single_site_potentials:
                    single_site_logits = self.single_site_head(results['embeddings'])
                    results['single_site_logits'] = single_site_logits

                # Compute single site potential alignment loss (always use dual head loss)
                if (self.use_single_site_potentials and single_site_potentials is not None and 'single_site_logits' in results):
                    single_site_loss = self.dual_head_loss_for_single_site(
                        logits=results['single_site_logits'],
                        dca_single_site_potentials=single_site_potentials,
                        tokens=tokens,
                        valid_residue_indices=valid_residue_indices,  # 传递已计算的valid_residue_indices
                        temperature=1.0,
                        reduction="mean"
                    )
                    results['single_site_loss'] = single_site_loss


                # logging.info("Applied DCA coupling bias injection to ESM attention layers")
            finally:
                self.remove_dca_bias(preserve_scaling=True) 
        # ======================= Bottom@bias injection ===========================================================================#


        # Default case - standard forward pass without bias or auxiliary losses
        else:
            output = self.esm_model(
                tokens,
                repr_layers=requested_layers,
                need_head_weights=False
            )
            
            results['embeddings'] = output['representations'][self.num_layers]
            
            if repr_layers:
                results['representations'] = {
                    layer: output['representations'][layer] 
                    for layer in repr_layers 
                    if layer in output['representations']
                }

        if self.training:
            # 只在训练模式下计算范围惩罚
            range_penalties = self.compute_range_penalties()
            if range_penalties['total_penalty'] > 0:  # 只有在有实际惩罚时才添加到结果中
                results['range_penalties'] = range_penalties
                results['range_penalty_loss'] = range_penalties['total_penalty']  # Weight applied at training level
        return results



class EvolutionaryScaleModelingCombined(EvolutionaryScaleModeling):
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        pretrained_weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        unfreeze_layers: Optional[List[int]] = None,
        use_auxiliary_losses: bool = True, # Must be True for this class
        use_relative_bias_injection: bool = True, # Must be True for this class
        use_concat_map: bool = False,
        use_single_site_potentials: bool = False,
        alignment_loss_mode: str = "kl",
        alignment_layers: Optional[List[int]] = None,
        attention_fusion_method: str = "weighted_average",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):

        # Ensure that the core features of this combined class are enabled.
        if not use_auxiliary_losses or not use_relative_bias_injection:
            raise ValueError(
                "EvolutionaryScaleModelingCombined requires both 'use_auxiliary_losses' "
                "and 'use_relative_bias_injection' to be True."
            )

        super().__init__(
            model_name=model_name,
            pretrained_weights_path=pretrained_weights_path,
            freeze_backbone=freeze_backbone,
            unfreeze_layers=unfreeze_layers,
            use_auxiliary_losses=use_auxiliary_losses,
            use_relative_bias_injection=use_relative_bias_injection,
            use_concat_map=use_concat_map,
            use_single_site_potentials=use_single_site_potentials,
            alignment_loss_mode=alignment_loss_mode,
            alignment_layers=alignment_layers,
            attention_fusion_method=attention_fusion_method,
            device=device
        )
        
        logging.info(f"✓ Initialized EvoSiteCombined model with:")
        logging.info(f"  - Model: {model_name}")
        logging.info(f"  - Bias injection: {self.use_relative_bias_injection}")
        logging.info(f"  - Auxiliary losses: {self.use_auxiliary_losses}")
        logging.info(f"  - Auxiliary loss mode: {'contact maps' if use_concat_map else 'attention maps'}")



    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        sequences: Optional[List[str]] = None,
        dca_coupling_matrix: Optional[torch.Tensor] = None,
        single_site_potentials: Optional[torch.Tensor] = None,
        repr_layers: Optional[List[int]] = None,
        dynamic_scaling_factor: Optional[float] = None
    ) -> Dict[str, Any]:

        # ============================= PRIMARY CASE: Combined Bias Injection + Auxiliary Losses =============================
        if dca_coupling_matrix is not None and self.use_relative_bias_injection and self.use_auxiliary_losses:
            
            # --- Setup: Tokenization and Masking ---
            if tokens is None and sequences is not None:
                tokens = self.sequence_encoder.encode_batch(sequences).to(self._device)
            elif tokens is None:
                raise ValueError("Either tokens or sequences must be provided")
            tokens = tokens.to(self._device)
            batch_size, seq_len = tokens.shape

            valid_tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
            valid_tokens_mask &= (tokens != self.alphabet.padding_idx)
            if self.alphabet.prepend_bos: valid_tokens_mask &= (tokens != self.alphabet.cls_idx)
            if self.alphabet.append_eos: valid_tokens_mask &= (tokens != self.alphabet.eos_idx)
            if hasattr(self.alphabet, 'mask_idx'): valid_tokens_mask &= (tokens != self.alphabet.mask_idx)
            if hasattr(self.alphabet, 'unk_idx'): valid_tokens_mask &= (tokens != self.alphabet.unk_idx)
            
            valid_residue_indices = [torch.nonzero(mask, as_tuple=True)[0] for mask in valid_tokens_mask]

            # --- Setup: Layer Representation Extraction ---
            if repr_layers is None: repr_layers = []
            requested_layers = list(set(repr_layers))
            if self.num_layers not in requested_layers: requested_layers.append(self.num_layers)
            
            results = {
                'valid_tokens_mask': valid_tokens_mask,
                'valid_residue_indices': valid_residue_indices
            }

            # --- Core Logic: Bias Injection and Forward Pass ---
            coupling_matrices = dca_coupling_matrix if isinstance(dca_coupling_matrix, list) else [dca_coupling_matrix]
            
            try:
                self.remove_dca_bias()
                scaling_factor = dynamic_scaling_factor if dynamic_scaling_factor is not None else 1.0
                self.inject_dca_bias(dca_coupling_matrices=coupling_matrices, scaling_factor=scaling_factor)
                
                # Perform forward pass with bias active. We must request head weights for the auxiliary loss.
                output = self.esm_model(
                    tokens,
                    repr_layers=requested_layers,
                    return_contacts=self.use_concat_map,
                    need_head_weights=not self.use_concat_map
                )
            finally:
                self.remove_dca_bias(preserve_scaling=True)
            
            # --- Process Outputs and Compute Losses ---
            results['embeddings'] = output['representations'][self.num_layers]
            if repr_layers:
                results['representations'] = {
                    layer: output['representations'][layer] for layer in repr_layers if layer in output['representations']
                }

            if self.use_single_site_potentials:
                single_site_logits = self.single_site_head(results['embeddings'])
                results['single_site_logits'] = single_site_logits
                if single_site_potentials is not None:
                    results['single_site_loss'] = self.dual_head_loss_for_single_site(
                        logits=results['single_site_logits'],
                        dca_single_site_potentials=single_site_potentials,
                        valid_residue_indices=valid_residue_indices,
                    )
            
            # --- Auxiliary Loss Calculation (using efficient, vectorized parent logic) ---
            if self.use_concat_map:
                if 'contacts' in output and output['contacts'] is not None:
                    results['contact_maps'] = output['contacts']
                    results['coupling_losses'] = self.compute_coupling_alignment_loss(
                        contact_maps=results['contact_maps'],
                        dca_coupling_matrix=dca_coupling_matrix,
                        mode=self.alignment_loss_mode
                    )
                else:
                    raise RuntimeError("Cannot compute auxiliary loss: 'contacts' not found in model output.")
            else: # Use attention maps
                if 'attentions' in output and output['attentions'] is not None:
                    # REUSE of efficient parent logic for attention processing
                    esm_attentions = output['attentions']
                    layer_indices_tensor = torch.tensor(self.alignment_layers, device=esm_attentions.device)
                    selected_attentions = esm_attentions.index_select(1, layer_indices_tensor)
                    avg_head_attentions = selected_attentions.mean(dim=2)

                    attention_maps_by_layer = {}
                    for i, layer_idx in enumerate(self.alignment_layers):
                        layer_maps = []
                        for b in range(batch_size):
                            valid_idx = valid_residue_indices[b]
                            if valid_idx.numel() > 0:
                                valid_map = avg_head_attentions[b, i, valid_idx[:, None], valid_idx]
                                layer_maps.append(valid_map)
                            else:
                                layer_maps.append(torch.empty(0, 0, device=tokens.device, dtype=esm_attentions.dtype))
                        attention_maps_by_layer[layer_idx] = layer_maps
                    
                    results['attention_maps'] = self._fuse_attention_layers(
                        attention_maps_by_layer, fusion_method=self.attention_fusion_method
                    )
                    results['coupling_losses'] = self.compute_coupling_alignment_loss(
                        attention_maps=results['attention_maps'],
                        dca_coupling_matrix=dca_coupling_matrix,
                        mode=self.alignment_loss_mode
                    )
                else:
                     raise RuntimeError("Cannot compute auxiliary loss: 'attentions' not found in model output.")
            
            # Range penalty calculation
            if self.training:
                range_penalties = self.compute_range_penalties()
                if range_penalties['total_penalty'] > 0:
                    results['range_penalties'] = range_penalties
                    results['range_penalty_loss'] = range_penalties['total_penalty']
            
            return results

        # ============================= FALLBACK CASE: Not combined mode =============================
        else:
            raise ValueError("Either dca_coupling_matrix or self.use_relative_bias_injection must be provided")

