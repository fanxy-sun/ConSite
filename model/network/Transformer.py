import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from model.network.performer_pytorch import SelfAttention

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# for gradient checkpointing
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2*(x-mean)
        x /= std
        x += self.b_2
        return x

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, p_drop=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p_drop, inplace=True)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, src):
        src = self.linear2(self.dropout(F.relu_(self.linear1(src))))
        return src

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        if k_dim == None:
            k_dim = d_model
        if v_dim == None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scaling = 1/math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, mask=None, return_att=False):
        batch, L1 = query.shape[:2]
        batch, L2 = key.shape[:2]
        q = self.to_query(query).view(batch, L1, self.heads, self.d_k).permute(0,2,1,3) # (B, h, L, d_k)
        k = self.to_key(key).view(batch, L2, self.heads, self.d_k).permute(0,2,1,3) # (B, h, L, d_k)
        v = self.to_value(value).view(batch, L2, self.heads, self.d_k).permute(0,2,1,3)
        #
        attention = torch.matmul(q, k.transpose(-2, -1))*self.scaling

        # 在softmax之前应用掩码
        if mask is not None:
            attention = attention.masked_fill(mask == 0, torch.finfo(attention.dtype).min)

        attention = F.softmax(attention, dim=-1) # (B, h, L1, L2)
        attention = self.dropout(attention)
        #
        out = torch.matmul(attention, v) # (B, h, L, d_k)
        out = out.permute(0,2,1,3).contiguous().view(batch, L1, -1)
        #
        out = self.to_out(out)
        if return_att:
            attention = 0.5*(attention + attention.permute(0,1,3,2))
            return out, attention.permute(0,2,3,1)
        return out

# Own implementation for tied multihead attention
class TiedMultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(TiedMultiheadAttention, self).__init__()
        if k_dim == None:
            k_dim = d_model
        if v_dim == None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scaling = 1/math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, return_att=False):
        B, N, L = query.shape[:3]
        q = self.to_query(query).view(B, N, L, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, N, h, l, k)
        k = self.to_key(key).view(B, N, L, self.heads, self.d_k).permute(0,1,3,4,2).contiguous() # (B, N, h, k, l)
        v = self.to_value(value).view(B, N, L, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, N, h, l, k)
        #
        #attention = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(N*self.d_k) # (B, N, h, L, L)
        #attention = attention.sum(dim=1) # tied attention (B, h, L, L)
        scale = self.scaling / math.sqrt(N)
        q = q * scale
        attention = torch.einsum('bnhik,bnhkj->bhij', q, k)
        attention = F.softmax(attention, dim=-1) # (B, h, L, L)
        attention = self.dropout(attention)
        attention = attention.unsqueeze(1) # (B, 1, h, L, L)
        #
        out = torch.matmul(attention, v) # (B, N, h, L, d_k)
        out = out.permute(0,1,3,2,4).contiguous().view(B, N, L, -1)
        #
        out = self.to_out(out)
        if return_att:
            attention = attention.squeeze(1)
            attention = 0.5*(attention + attention.permute(0,1,3,2))
            attention = attention.permute(0,3,1,2)
            return out, attention
        return out

class SequenceWeight(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(SequenceWeight, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, msa, msa_mask):
        B, N, L = msa.shape[:3]
        
        msa = msa.permute(0,2,1,3) # (B, L, N, K)
        tar_seq = msa[:,:,0].unsqueeze(2) # (B, L, 1, K)
        
        q = self.to_query(tar_seq).view(B, L, 1, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, L, h, 1, k)
        k = self.to_key(msa).view(B, L, N, self.heads, self.d_k).permute(0,1,3,4,2).contiguous() # (B, L, h, k, N)
        
        q = q * self.scale
        attn = torch.matmul(q, k) # (B, L, h, 1, N)

        if msa_mask is not None:
            # msa_mask (B, N) -> (B, 1, 1, 1, N)
            mask = msa_mask.view(B, 1, 1, 1, N)
            attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)

        attn = F.softmax(attn, dim=-1)
        return self.dropout(attn)

# Own implementation for multihead attention (Input shape: Batch, Len, Emb)
class SoftTiedMultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(SoftTiedMultiheadAttention, self).__init__()
        if k_dim == None:
            k_dim = d_model
        if v_dim == None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.seq_weight = SequenceWeight(d_model, heads, dropout=dropout)
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, seq_mask=None, msa_mask=None, return_att=False):
        B, N, L = query.shape[:3]
        # SequenceWeight 需要 msa_mask 来正确地基于 MSA 深度为序列加权
        seq_weight = self.seq_weight(query, msa_mask=msa_mask) # (B, L, h, 1, N)
        seq_weight = seq_weight.permute(0,4,2,1,3) # (B, N, h, l, -1)
        #
        q = self.to_query(query).view(B, N, L, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, N, h, l, k)
        k = self.to_key(key).view(B, N, L, self.heads, self.d_k).permute(0,1,3,4,2).contiguous() # (B, N, h, k, l)
        v = self.to_value(value).view(B, N, L, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, N, h, l, k)
        #
        #attention = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(N*self.d_k) # (B, N, h, L, L)
        #attention = attention.sum(dim=1) # tied attention (B, h, L, L)
        q = q * seq_weight # (B, N, h, l, k)
        k = k * self.scale
        attention = torch.einsum('bnhik,bnhkj->bhij', q, k)

        # 在 softmax 前应用序列掩码，以防止对填充位置的注意力
        if seq_mask is not None:
            # seq_mask (B, L) -> (B, 1, L, L)，以便与 attention map (B, h, L, L) 进行广播
            mask = seq_mask.unsqueeze(1) * seq_mask.unsqueeze(2)
            attention = attention.masked_fill(mask.unsqueeze(1) == 0, torch.finfo(attention.dtype).min)
        attention = F.softmax(attention, dim=-1) # (B, h, L, L)
        attention = self.dropout(attention)
        del q, k, seq_weight
        #
        #out = torch.matmul(attention, v) # (B, N, h, L, d_k)
        out = torch.einsum('bhij,bnhjk->bnhik', attention, v)
        out = out.permute(0,1,3,2,4).contiguous().view(B, N, L, -1)
        #
        out = self.to_out(out)
        
        if return_att:
            attention = 0.5*(attention + attention.permute(0,1,3,2))
            attention = attention.permute(0,2,3,1) # (B, L, L, h)
            return out, attention
        return out

class DirectMultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, heads, dropout=0.1):
        super(DirectMultiheadAttention, self).__init__()
        self.heads = heads
        self.proj_pair = nn.Linear(d_in, heads)
        self.drop = nn.Dropout(dropout, inplace=True)
        # linear projection to get values from given msa
        self.proj_msa = nn.Linear(d_out, d_out)
        # projection after applying attention
        self.proj_out = nn.Linear(d_out, d_out)
    
    def forward(self, src, tgt, mask=None):
        B, N, L = tgt.shape[:3]
        attn_logits = self.proj_pair(src) # (B, L, L, h)

        if mask is not None:
            # mask (B, L, L) -> (B, L, L, 1)
            mask = mask.unsqueeze(-1)
            attn_logits = attn_logits.masked_fill(mask == 0, torch.finfo(attn_logits.dtype).min)

        attn_map = F.softmax(attn_logits, dim=2).permute(0,3,1,2) # (B, h, L, L)
        attn_map = self.drop(attn_map).unsqueeze(1)
        
        # apply attention
        value = self.proj_msa(tgt).permute(0,3,1,2).contiguous().view(B, -1, self.heads, N, L) # (B,-1, h, N, L)
        tgt = torch.matmul(value, attn_map).view(B, -1, N, L).permute(0,2,3,1) # (B,N,L,K)
        tgt = self.proj_out(tgt)
        return tgt

class MaskedDirectMultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, heads, d_k=32, dropout=0.1):
        super(MaskedDirectMultiheadAttention, self).__init__()
        self.heads = heads
        self.scaling = 1/math.sqrt(d_k)
        
        self.to_query = nn.Linear(d_in, heads*d_k)
        self.to_key   = nn.Linear(d_in, heads*d_k)
        self.to_value = nn.Linear(d_out, d_out)
        self.to_out   = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, mask):
        batch, N, L = value.shape[:3] 
        #
        # project to query, key, value
        q = self.to_query(query).view(batch, L, self.heads, -1).permute(0,2,1,3) # (B, h, L, -1)
        k = self.to_key(key).view(batch, L, self.heads, -1).permute(0,2,1,3) # (B, h, L, -1)
        v = self.to_value(value).view(batch, N, L, self.heads, -1).permute(0,3,1,2,4) # (B, h, N, L, -1)
        #
        q = q*self.scaling
        attention = torch.matmul(q, k.transpose(-2, -1)) # (B, h, L, L)
        attention = attention.masked_fill(mask < 0.5, torch.finfo(q.dtype).min)
        attention = F.softmax(attention, dim=-1) # (B, h, L1, L2)
        attention = self.dropout(attention) # (B, h, 1, L, L)
        #
        #out = torch.matmul(attention, v) # (B, h, N, L, d_out//h)
        out = torch.einsum('bhij,bhnjk->bhnik', attention, v) # (B, h, N, L, d_out//h)
        out = out.permute(0,2,3,1,4).contiguous().view(batch, N, L, -1)
        #
        out = self.to_out(out)
        return out

# Use PreLayerNorm for more stable training
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, p_drop=0.1, performer_opts=None, use_tied=False):
        super(EncoderLayer, self).__init__()
        self.use_performer = performer_opts is not None
        self.use_tied = use_tied
        # multihead attention
        if self.use_performer:
            self.attn = SelfAttention(dim=d_model, heads=heads, dropout=p_drop, 
                                      generalized_attention=True, **performer_opts)
        elif use_tied:
            self.attn = SoftTiedMultiheadAttention(d_model, heads, dropout=p_drop)
        else:
            self.attn = MultiheadAttention(d_model, heads, dropout=p_drop)
        # feedforward
        self.ff = FeedForwardLayer(d_model, d_ff, p_drop=p_drop)

        # normalization module
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop, inplace=True)
        self.dropout2 = nn.Dropout(p_drop, inplace=True)

    # def forward(self, src, seq_mask=None, msa_mask=None, return_att=False):
    #     """
    #     Args:
    #         src (torch.Tensor): 输入MSA特征, shape (B, N, L, d_msa)
    #         seq_mask (torch.Tensor, optional): L维度的掩码, shape (B, L).
    #         msa_mask (torch.Tensor, optional): N维度的掩码, shape (B, N).
    #     """
    #     # multihead attention w/ pre-LayerNorm
    #     B, N, L = src.shape[:3]
    #     src2 = self.norm1(src)
    #     if not self.use_tied:
    #         src2 = src2.reshape(B*N, L, -1)
    #     if return_att:
    #         src2, att = self.attn(src2, src2, src2, seq_mask=None, msa_mask=None, return_att=return_att)
    #         src2 = src2.reshape(B,N,L,-1)
    #     else:
    #         src2 = self.attn(src2, src2, src2).reshape(B,N,L,-1)
    #     src = src + self.dropout1(src2)

    #     # feed-forward
    #     src2 = self.norm2(src) # pre-normalization
    #     src2 = self.ff(src2)
    #     src = src + self.dropout2(src2)
    #     if return_att:
    #         return src, att
    #     return src
    def forward(self, src, seq_mask=None, msa_mask=None, return_att=False):
        """
        Args:
            src (torch.Tensor): 输入特征, (B, N_dim, L_dim, d_model)或者(B, L_dim, N_dim, d_msa)
            seq_mask (torch.Tensor, optional): 对应 attention 维度 (L_dim) 的掩码, e.g., (B, L) for row attention.
            msa_mask (torch.Tensor, optional): MSA 深度掩码 (B, N), 仅用于 tied-attention.
            return_att (bool): 是否返回注意力图.
        """
        # multihead attention w/ pre-LayerNorm
        B, N_dim, L_dim = src.shape[:3]
        src2 = self.norm1(src)

        att_out = None
        if self.use_tied:
            # Tied attention (SoftTiedMultiheadAttention) 直接处理 (B, N, L, d) 输入
            # 并且需要 seq_mask (L维度) 和 msa_mask (N维度加权)
            if return_att:
                src2, att_out = self.attn(src2, src2, src2, seq_mask=seq_mask, msa_mask=msa_mask, return_att=return_att)
            else:
                src2 = self.attn(src2, src2, src2, seq_mask=seq_mask, msa_mask=msa_mask)
        else:
            # 标准 attention 需要重塑输入为 (Batch, Length, Dim)
            src2 = src2.reshape(B * N_dim, L_dim, -1)
            
            # 为重塑后的输入创建掩码
            mask = None
            if seq_mask is not None:
                # seq_mask 对应 L_dim。需要为新的批次维度 B*N_dim 进行重复
                mask_1d = seq_mask.repeat_interleave(N_dim, dim=0) # (B*N, L)
                if self.use_performer:
                    # Performer (SelfAttention) 直接使用一维的 padding mask
                    mask = mask_1d
                else:
                    # 标准 MultiheadAttention 需要一个可以广播到 (B, h, L, L) 的二维掩码
                    mask = (mask_1d.unsqueeze(1) * mask_1d.unsqueeze(2)).unsqueeze(1)
            
            # 不同注意力实现的掩码参数名不同
            if self.use_performer:
                src2 = self.attn(src2, src2, src2, padding_mask=mask)
            else:
                if return_att:
                    # MultiheadAttention 需要 mask 参数, 并且可以返回注意力图
                    src2, att_out = self.attn(src2, src2, src2, mask=mask, return_att=return_att)
                else:
                    src2 = self.attn(src2, src2, src2, mask=mask)
            
            src2 = src2.reshape(B, N_dim, L_dim, -1)

        src = src + self.dropout1(src2)

        # feed-forward
        src2 = self.norm2(src) # pre-normalization
        src2 = self.ff(src2)
        src = src + self.dropout2(src2)
        
        if return_att:
            return src, att_out
        return src

# AxialTransformer with tied attention for L dimension
class AxialEncoderLayer(nn.Module):
    """
    AxialEncoderLayer内的掩码策略只适用于self.attn_L和self.attn_N都是SelfAttention的情况
    """
    def __init__(self, d_model, d_ff, heads, p_drop=0.1, performer_opts=None,
                 use_tied_row=False, use_tied_col=False, use_soft_row=False):
        super(AxialEncoderLayer, self).__init__()
        self.use_performer = performer_opts is not None
        self.use_tied_row = use_tied_row
        self.use_tied_col = use_tied_col
        self.use_soft_row = use_soft_row
        # multihead attention
        if use_tied_row:
            self.attn_L = TiedMultiheadAttention(d_model, heads, dropout=p_drop)
        elif use_soft_row:
            self.attn_L = SoftTiedMultiheadAttention(d_model, heads, dropout=p_drop)
        else:
            if self.use_performer:
                self.attn_L = SelfAttention(dim=d_model, heads=heads, dropout=p_drop, 
                                            generalized_attention=True, **performer_opts)
            else:
                self.attn_L = MultiheadAttention(d_model, heads, dropout=p_drop)
        if use_tied_col:
            self.attn_N = TiedMultiheadAttention(d_model, heads, dropout=p_drop)
        else:
            if self.use_performer:
                self.attn_N = SelfAttention(dim=d_model, heads=heads, dropout=p_drop, 
                                            generalized_attention=True, **performer_opts)
            else:
                self.attn_N = MultiheadAttention(d_model, heads, dropout=p_drop)

        # feedforward
        self.ff = FeedForwardLayer(d_model, d_ff, p_drop=p_drop)

        # normalization module
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop, inplace=True)
        self.dropout2 = nn.Dropout(p_drop, inplace=True)
        self.dropout3 = nn.Dropout(p_drop, inplace=True)

    def forward(self, src, mask=None, return_att=False):
        """
        Args:
            src (torch.Tensor): 输入特征, shape (B, N, L, dim)
            mask (torch.Tensor, optional): 掩码, shape (B, L, L). Defaults to None.
        Returns:
            torch.Tensor: 输出特征, shape (B, N, L, dim)
        """
        # Input shape for multihead attention: (BATCH, NSEQ, NRES, EMB)
        # Tied multihead attention w/ pre-LayerNorm
        B, N, L = src.shape[:3]

        row_padding_mask, col_padding_mask = None, None
        if mask is not None:
            # mask 的形状是 (B, L, L)，是一个 pair_mask
            # 我们需要从中推断出 1D 的 padding_mask
            # 如果第 i 行/列全为 False，则位置 i 是 padding
            # .any(dim=-1) 检查每一行是否至少有一个 True
            padding_mask_L = mask.any(dim=-1) # 形状 (B, L)
            padding_mask_N = mask.any(dim=-2) # 形状 (B, L), 在 N==L 时等于 padding_mask_L
            
            # 为行注意力准备掩码 (作用于 (B*N, L) 张量)
            row_padding_mask = padding_mask_L.repeat_interleave(N, dim=0) # (B*N, L)
            
            # 为列注意力准备掩码 (作用于 (B*L, N) 张量)
            col_padding_mask = padding_mask_N.repeat_interleave(L, dim=0) # (B*L, N)

        src2 = self.norm1(src)
        if self.use_tied_row or self.use_soft_row:
            src2 = self.attn_L(src2, src2, src2) # Tied attention over L
        else:
            src2 = src2.reshape(B*N, L, -1)
            src2 = self.attn_L(src2, src2, src2, padding_mask=row_padding_mask)
            src2 = src2.reshape(B, N, L, -1)
        src = src + self.dropout1(src2)
        
        # attention over N
        src2 = self.norm2(src)
        if self.use_tied_col:
            src2 = src2.permute(0,2,1,3)
            src2 = self.attn_N(src2, src2, src2) # Tied attention over N
            src2 = src2.permute(0,2,1,3)
        else:
            src2 = src2.permute(0,2,1,3).reshape(B*L, N, -1)
            src2 = self.attn_N(src2, src2, src2, padding_mask=col_padding_mask) # attention over N
            src2 = src2.reshape(B, L, N, -1).permute(0,2,1,3)
        src = src + self.dropout2(src2)

        # feed-forward
        src2 = self.norm3(src) # pre-normalization
        src2 = self.ff(src2)
        src = src + self.dropout3(src2)
        return src

class Encoder(nn.Module):
    def __init__(self, enc_layer, n_layer):
        super(Encoder, self).__init__()
        self.layers = _get_clones(enc_layer, n_layer)
        self.n_layer = n_layer
   
    def forward(self, src, return_att=False, **kwargs):
        output = src
        for layer in self.layers:
            output = layer(output, return_att=return_att, **kwargs)
        return output

class CrossEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, d_k, d_v, performer_opts=None, p_drop=0.1):
        super(CrossEncoderLayer, self).__init__()
        self.use_performer = performer_opts is not None
        
        # multihead attention
        if self.use_performer:
            self.attn = SelfAttention(dim=d_model, k_dim=d_k, heads=heads, dropout=p_drop,
                                      generalized_attention=True, **performer_opts)
        else:
            self.attn = MultiheadAttention(d_model, heads, k_dim=d_k, v_dim=d_v, dropout=p_drop)
        # feedforward
        self.ff = FeedForwardLayer(d_model, d_ff, p_drop=p_drop)

        # normalization module
        self.norm = LayerNorm(d_k)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop, inplace=True)
        self.dropout2 = nn.Dropout(p_drop, inplace=True)

    def forward(self, src, tgt):
        # Input:
        #   For MSA to Pair: src (N, L, K), tgt (L, L, C)
        #   For Pair to MSA: src (L, L, C), tgt (N, L, K)
        # Input shape for multihead attention: (SRCLEN, BATCH, EMB)
        # multihead attention
        # pre-normalization
        src = self.norm(src)
        tgt2 = self.norm1(tgt)
        tgt2 = self.attn(tgt2, src, src) # projection to query, key, value are done in MultiheadAttention module
        tgt = tgt + self.dropout1(tgt2)
        
        # Feed forward
        tgt2 = self.norm2(tgt)
        tgt2 = self.ff(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        
        return tgt

class DirectEncoderLayer(nn.Module):
    def __init__(self, heads, d_in, d_out, d_ff, symmetrize=True, p_drop=0.1):
        super(DirectEncoderLayer, self).__init__()
        self.symmetrize = symmetrize

        self.attn = DirectMultiheadAttention(d_in, d_out, heads, dropout=p_drop)
        self.ff = FeedForwardLayer(d_out, d_ff, p_drop=p_drop)

        # dropouts
        self.drop_1 = nn.Dropout(p_drop, inplace=True)
        self.drop_2 = nn.Dropout(p_drop, inplace=True)
        # LayerNorm
        self.norm = LayerNorm(d_in)
        self.norm1 = LayerNorm(d_out)
        self.norm2 = LayerNorm(d_out)

    def forward(self, src, tgt, mask=None):
        # Input:
        #  For pair to msa: src=pair (B, L, L, C), tgt=msa (B, N, L, K)
        B, N, L = tgt.shape[:3]
        # get attention map
        if self.symmetrize:
            src = 0.5*(src + src.permute(0,2,1,3))
        src = self.norm(src)
        tgt2 = self.norm1(tgt)
        tgt2 = self.attn(src, tgt2, mask=mask)
        tgt = tgt + self.drop_1(tgt2)

        # feed-forward
        tgt2 = self.norm2(tgt.view(B*N,L,-1)).view(B,N,L,-1)
        tgt2 = self.ff(tgt2)
        tgt = tgt + self.drop_2(tgt2)

        return tgt

class CrossEncoder(nn.Module):
    def __init__(self, enc_layer, n_layer):
        super(CrossEncoder, self).__init__()
        self.layers = _get_clones(enc_layer, n_layer)
        self.n_layer = n_layer
    def forward(self, src, tgt, **kwargs):
        output = tgt
        for layer in self.layers:
            output = layer(src, output, **kwargs)
        return output


