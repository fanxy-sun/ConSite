import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.network.Transformer import EncoderLayer, AxialEncoderLayer, Encoder, LayerNorm

# Initial embeddings for target sequence, msa, template info
# positional encoding
#   option 1: using sin/cos --> using this for now 
#   option 2: learn positional embedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, p_drop=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.drop = nn.Dropout(p_drop,inplace=True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # (1, max_len, d_model)
    def forward(self, x, idx_s):
        """
        位置索引idx_s的掩码填充值建议为0
        Args:
            x: 输入的特征张量, 形状通常为 (B, N, L, dim)
            idx_s: 批次中每个序列的残基索引, 形状为 (B, L)
        Returns:
            torch.Tensor: 添加了位置编码的特征张量, (B, N, L, dim)
        """
        # 移除 for 循环，直接使用批量的索引张量 idx_s 进行索引
        # self.pe 的形状是 (1, max_len, d_model)
        # idx_s 的形状是 (B, L)
        # PyTorch的高级索引功能允许我们这样做：
        # - self.pe的第0维(大小为1)会自动广播以匹配idx_s的第0维(大小为B)
        # - self.pe的第1维(max_len)被idx_s中的值索引
        # - self.pe的第2维(d_model)保持不变
        # 结果 pe 的形状为 (B, L, d_model)
        pe = self.pe[0, idx_s]
        
        # 为了能够正确地与 MSA 特征 (B, N, L, d_model) 进行广播相加，
        # 我们需要在维度1上增加一个维度。
        # (B, L, d_model) -> (B, 1, L, d_model)
        # 这样它就可以和 (B, N, L, d_model) 的 x 相加了。
        # 如果 x 的形状是 (B, L, d_model)，广播机制同样适用。
        pe = pe.unsqueeze(1)
        
        # 原始代码中的 torch.autograd.Variable 是旧版PyTorch的用法，现在已不再需要。
        # 直接相加即可。因为 pe 是从 buffer 创建的，它本身就不带梯度。
        x = x + pe
        return self.drop(x)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.drop = nn.Dropout(p_drop,inplace=True)
        #
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) *
                             -(math.log(10000.0) / d_model_half))
        self.register_buffer('div_term', div_term)
    
    def forward(self, x, idx_s):
        """
        Args:
            x (torch.Tensor): 输入的2D特征张量。形状为 (B, L, L, K) 或 (B*T, L, L, K)。
            idx_s (torch.Tensor): 批次中每个序列的残基索引。形状为 (B_idx, L)。
        """
        B, L, _, K = x.shape
        B_idx = idx_s.shape[0]
        K_half = K // 2
        
        # --- 核心改造部分 (显存优化) ---
        
        # 1. 计算所有样本的1D位置编码
        # sin_inp 形状: (B_idx, L, K_half//2)
        sin_inp = torch.einsum('bi,d->bid', idx_s.float(), self.div_term)
        
        # emb 形状: (B_idx, L, K_half)
        emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
        del sin_inp # 及时释放中间变量

        # 2. 处理 x 和 idx_s 批次维度不匹配的情况
        # T (num_repeats) 是模板数量或者1
        num_repeats = 1
        if B % B_idx == 0:
            num_repeats = B // B_idx

        # 3. 分步计算、添加位置编码，并及时删除中间变量
        
        # --- Part 1: 添加列编码 (作用于前半部分特征) ---
        # emb_col 形状: (B_idx, L, 1, K_half)
        emb_col = emb.unsqueeze(2)
        
        if num_repeats > 1:
            # 扩展 emb_col 以匹配 x 的批次大小
            # (B_idx, L, 1, K_half) -> (B_idx, 1, L, 1, K_half) -> (B, L, 1, K_half)
            emb_col_expanded = emb_col.unsqueeze(1).repeat(1, num_repeats, 1, 1, 1).view(B, L, 1, K_half)
            # 使用 inplace 加法直接修改 x 的切片，避免创建大的中间张量
            x[:, :, :, :K_half].add_(emb_col_expanded)
            del emb_col_expanded # 及时释放
        else: # B == B_idx 的情况
            x[:, :, :, :K_half].add_(emb_col)
        
        del emb_col # 及时释放

        # --- Part 2: 添加行编码 (作用于后半部分特征) ---
        # emb_row 形状: (B_idx, 1, L, K_half)
        emb_row = emb.unsqueeze(1)
        
        if num_repeats > 1:
            # 扩展 emb_row 以匹配 x 的批次大小
            # (B_idx, 1, L, K_half) -> (B_idx, 1, 1, L, K_half) -> (B, 1, L, K_half)
            emb_row_expanded = emb_row.unsqueeze(1).repeat(1, num_repeats, 1, 1, 1).view(B, 1, L, K_half)
            x[:, :, :, K_half:].add_(emb_row_expanded)
            del emb_row_expanded
        else: # B == B_idx 的情况
            x[:, :, :, K_half:].add_(emb_row)

        del emb_row, emb # 释放最后剩余的中间变量

        return self.drop(x)

class QueryEncoding(nn.Module):
    def __init__(self, d_model):
        super(QueryEncoding, self).__init__()
        self.pe = nn.Embedding(2, d_model) # (0 for query, 1 for others)
    
    def forward(self, x):
        B, N, L, K = x.shape
        idx = torch.ones((B, N, L), device=x.device).long()
        idx[:,0,:] = 0 # first sequence is the query
        x = x + self.pe(idx)
        return x 

class MSA_emb(nn.Module):
    def __init__(self, d_model=64, d_msa=21, p_drop=0.1, max_len=5000):
        super(MSA_emb, self).__init__()
        self.emb = nn.Embedding(d_msa, d_model)
        self.pos = PositionalEncoding(d_model, p_drop=p_drop, max_len=max_len)
        self.pos_q = QueryEncoding(d_model)
    def forward(self, msa, idx):
        """
        Args:
            msa (torch.Tensor): MSA 序列，每个元素是氨基酸/gap 索引 (0-20)。形状 (B, N, L)
            idx (torch.Tensor): 残基的绝对位置索引。形状 (B, L)
        Returns:
            torch.Tensor: 经过嵌入、位置编码和查询编码的 MSA 特征。形状 (B, N, L, d_model)
        """
        B, N, L = msa.shape
        out = self.emb(msa) # (B, N, L, d_model)
        out = self.pos(out, idx) # add positional encoding
        return self.pos_q(out) # add query encoding

# pixel-wise attention based embedding (from trRosetta-tbm)
class Templ_emb(nn.Module):
    def __init__(self, d_t1d=3, d_t2d=10, d_templ=64, n_att_head=4, r_ff=4,
                 performer_opts=None, p_drop=0.1, max_len=5000):
        super(Templ_emb, self).__init__()
        self.proj = nn.Linear(d_t1d*2+d_t2d+1, d_templ)
        self.pos = PositionalEncoding2D(d_templ, p_drop=p_drop)
        # attention along L
        enc_layer_L = AxialEncoderLayer(d_templ, d_templ*r_ff, n_att_head, p_drop=p_drop,
                                        performer_opts=performer_opts)
        self.encoder_L = Encoder(enc_layer_L, 1)
        
        self.norm = LayerNorm(d_templ)
        self.to_attn = nn.Linear(d_templ, 1)

    def forward(self, t1d, t2d, idx, seq_mask):
        """
        Args:
            t1d (torch.Tensor): 1D模板特征, shape (B, T, L, d_t1d)
            t2d (torch.Tensor): 2D模板特征, shape (B, T, L, L, d_t2d)
            seq_mask (torch.Tensor): 序列长度掩码, shape (B, L)
        """
        B, T, L, _ = t1d.shape
        left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
        right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)
        seqsep = torch.abs(idx[:,:,None]-idx[:,None,:]) + 1
        seqsep = torch.log(seqsep.float()).view(B,L,L,1).unsqueeze(1).expand(-1,T,-1,-1,-1)
        #
        feat = torch.cat((t2d, left, right, seqsep), -1)
        del left, right, seqsep # 优化显存：及时删除中间变量

        # self.proj(feat) 形状: (B, T, L, L, d_templ)
        # .reshape(B*T, L, L, -1) 形状: (B*T, L, L, d_templ)
        feat = self.proj(feat).reshape(B*T, L, L, -1)
        feat = self.pos(feat, idx) # add positional embedding, feat 形状 (B*T, L, L, d_templ)

        # 创建并传递 pair_mask. pair_mask 形状: (B, L, L)
        pair_mask = seq_mask.unsqueeze(1) * seq_mask.unsqueeze(2)
        pair_mask = pair_mask.repeat(T, 1, 1) # 形状变为 (B*T, L, L)

        # 移除 for 循环，将 B*T 维度作为批次维度直接传入 encoder
        # self.encoder_L 期望的输入形状是 (批次, ...)，这里批次是 B*T
        # AxialEncoderLayer 内部会处理 (B*T, L, L, d_templ) 这种输入
        feat = self.encoder_L(feat, mask=pair_mask)

        feat = feat.reshape(B, T, L, L, -1)
        feat = feat.permute(0,2,3,1,4).contiguous().reshape(B, L*L, T, -1)

        # 将填充像素的特征置零
        pixel_mask = pair_mask.reshape(B, L*L) # 形状: (B, L*L)
        feat = feat * pixel_mask.unsqueeze(-1).unsqueeze(-1)

        # attn 形状 (B, L*L, T, 1)
        # pixel_mask 形状 (B, L*L) -> (B, L*L, 1, 1)
        attn = self.to_attn(self.norm(feat))
        attn = attn.masked_fill(pixel_mask.unsqueeze(-1).unsqueeze(-1) == 0, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-2) # (B, L*L, T, 1)
        feat = torch.matmul(attn.transpose(-2, -1), feat)
        return feat.reshape(B, L, L, -1)

class Pair_emb_w_templ(nn.Module):
    def __init__(self, d_model=128, d_seq=21, d_templ=64, p_drop=0.1):
        super(Pair_emb_w_templ, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.norm_templ = LayerNorm(d_templ)
        self.projection = nn.Linear(d_model + d_templ + 1, d_model)
        self.pos = PositionalEncoding2D(d_model, p_drop=p_drop)

    def forward(self, seq, idx, templ):
        """
        Args:
            seq (torch.Tensor): 目标序列, shape (B, L)
            idx (torch.Tensor): 残基索引, shape (B, L)
            templ (torch.Tensor): 2D模板特征, shape (B, L, L, d_templ)
        Returns:
            torch.Tensor: 经过嵌入、位置编码的2D对特征， shape (B, L, L, d_model)
        """
        B = seq.shape[0]
        L = seq.shape[1]
        #
        # get initial sequence pair features
        seq = self.emb(seq) # (B, L, d_model//2)
        left  = seq.unsqueeze(2).expand(-1,-1,L,-1)
        right = seq.unsqueeze(1).expand(-1,L,-1,-1)
        seqsep = torch.abs(idx[:,:,None]-idx[:,None,:])+1 
        seqsep = torch.log(seqsep.float()).view(B,L,L,1)
        #
        templ = self.norm_templ(templ)
        pair = torch.cat((left, right, seqsep, templ), dim=-1)
        pair = self.projection(pair) # (B, L, L, d_model)
        
        return self.pos(pair, idx)

class Pair_emb_wo_templ(nn.Module):
    #TODO: embedding without template info
    def __init__(self, d_model=128, d_seq=21, p_drop=0.1):
        super(Pair_emb_wo_templ, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.projection = nn.Linear(d_model + 1, d_model)
        self.pos = PositionalEncoding2D(d_model, p_drop=p_drop)
    def forward(self, seq, idx):
        # input:
        #   seq: target sequence (B, L, 20)
        B = seq.shape[0]
        L = seq.shape[1]
        seq = self.emb(seq) # (B, L, d_model//2)
        left  = seq.unsqueeze(2).expand(-1,-1,L,-1)
        right = seq.unsqueeze(1).expand(-1,L,-1,-1)
        seqsep = torch.abs(idx[:,:,None]-idx[:,None,:])+1 
        seqsep = torch.log(seqsep.float()).view(B,L,L,1)
        #
        pair = torch.cat((left, right, seqsep), dim=-1)
        pair = self.projection(pair)
        return self.pos(pair, idx)

