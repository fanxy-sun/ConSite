import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import torch.utils.checkpoint as checkpoint
from model.rfdiffusion.util import get_tips
from model.rfdiffusion.util_module import Dropout, create_custom_forward, rbf, init_lecun_normal
from model.rfdiffusion.Attention_module import Attention, FeedForwardLayer, AttentionWithBias
from model.rfdiffusion.Track_module import PairStr2Pair
import math
import numpy as np 

# Module contains classes and functions to generate initial embeddings

class PositionalEncoding2D(nn.Module):
    """
    $$ AI编写，未审查
    """
    # Add relative positional encoding to pair features
    def __init__(self, d_model, minpos=-32, maxpos=32, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos)+maxpos+1
        self.emb = nn.Embedding(self.nbin, d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x, idx, cyclize=None, attention_mask=None):
        # 仅在 cyclize=None 场景下扩展批量支持，保持原始分桶逻辑完全一致
        squeeze_x = False
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_x = True

        B, L = idx.shape
        seqsep = idx[:, None, :] - idx[:, :, None]  # (B, L, L)

        bins = torch.arange(self.minpos, self.maxpos, device=idx.device)
        ib = torch.bucketize(seqsep.reshape(B * L * L), bins).view(B, L, L)

        emb = self.emb(ib)  # (B, L, L, d_model)

        if attention_mask is not None:
            pair_mask = attention_mask[:, :, None] & attention_mask[:, None, :]
            # 将填充对的位置编码设为0
            emb = emb * pair_mask.unsqueeze(-1)

        x = self.drop(x + emb)

        if squeeze_x:
            x = x.squeeze(0)
        return x

class MSA_emb(nn.Module):
    """
    MSA_emb 是 RFdiffusion/RoseTTAFold 体系中用于生成初始 MSA（多序列比对）和 pair（残基对）特征的嵌入模块。它将原始 MSA、主序列、残基索引等信息编码为后续网络处理的高维特征。
    """
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_pair=128, d_state=32, d_init=22+22+2+2,
                 minpos=-32, maxpos=32, p_drop=0.1, input_seq_onehot=False):
        """
        Args:
            d_msa：MSA嵌入的维度，默认256。
            d_pair：pair嵌入的维度，默认128。
            d_state：state嵌入的维度，默认32。
            d_init：MSA输入的最后一维，默认48（22+22+2+2）。
            minpos/maxpos：相对位置编码的范围。
        """
        super(MSA_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(22, d_msa) # embedding for query sequence -- used for MSA embedding

        # 在构建 pair 特征时，网络需要对每一对残基 (i, j) 生成一个嵌入。
        # emb_left 用于编码第 i 个残基的氨基酸类型，emb_right 用于编码第 j 个残基的氨基酸类型。
        # 这样，pair(i, j) 的嵌入就是 emb_left(seq[i]) + emb_right(seq[j])，即分别对左、右残基做嵌入，然后相加。
        self.emb_left = nn.Embedding(22, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_right = nn.Embedding(22, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_state = nn.Embedding(22, d_state)
        self.drop = nn.Dropout(p_drop)
        self.pos = PositionalEncoding2D(d_pair, minpos=minpos, maxpos=maxpos, p_drop=p_drop)

        self.input_seq_onehot=input_seq_onehot

        self.reset_parameter()

    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        self.emb_q = init_lecun_normal(self.emb_q)
        self.emb_left = init_lecun_normal(self.emb_left)
        self.emb_right = init_lecun_normal(self.emb_right)
        self.emb_state = init_lecun_normal(self.emb_state)

        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx, cyclize, attention_mask=None):
        """
        Inputs:
          - msa: Input MSA (B, N, L, d_init)
          - seq: Input Sequence (B, L, 22)
          - idx: Residue index (B, L)
        Outputs:
          - msa: Initial MSA embedding (B, N, L, d_msa)
          - pair: Initial Pair embedding (B, L, L, d_pair)
          - state: Initial State embedding (B, L, d_state)
                * 在 RoseTTAFold/RFdiffusion 的三轨架构（MSA轨道、Pair轨道、Structure轨道）中，state 是连接`序列/配对信息`与`三维结构信息`的关键桥梁。
                * MSA轨道 (msa) 和 Pair轨道 (pair) 主要在二维空间中处理序列和共进化信息。Structure轨道 则在三维空间中通过 SE(3)-Transformer 更新原子坐标。SE(3)-Transformer 需要为每个残基（图中的节点）提供一个初始的特征向量，state 正是扮演了这个初始节点特征的角色。
                * state最初完全由输入序列 seq 生成。state 会被送入 IterativeSimulator 中的 IterBlock，并最终传递给 Str2Str 模块（SE(3)-Transformer），作为结构更新的输入节点特征。SE(3)-Transformer 在更新坐标的同时，也会更新 state 特征，使其包含丰富的局部三维环境信息。更新后的 state 会在下一次迭代中被用来增强 MSA 和 Pair 特征（例如，在 MSAPairStr2MSA 模块中，state 的信息会被加到 MSA 特征上）
        """


        N = msa.shape[1] # number of sequenes in MSA

        # msa embedding
        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding

        # Sergey's one hot trick
        tmp = (seq @ self.emb_q.weight).unsqueeze(1) # (B, 1, L, d_model) -- query embedding

        # 每条MSA序列都加上主序列的嵌入，强化主序列信息
        msa = msa + tmp.expand(-1, N, -1, -1) # adding query embedding to MSA
        msa = self.drop(msa)

        # pair embedding
        # Sergey's one hot trick
        left  = (seq @ self.emb_left.weight)[:,None] # (B, 1, L, d_pair)
        right = (seq @ self.emb_right.weight)[:,:,None] # (B, L, 1, d_pair)

        pair = left + right # (B, L, L, d_pair)
        pair = self.pos(pair, idx, cyclize, attention_mask=attention_mask) # add relative position

        # state embedding
        # Sergey's one hot trick
        state = self.drop(seq @ self.emb_state.weight)
        return msa, pair, state

class Extra_emb(nn.Module):
    """
    Extra_emb 是用于生成“额外”MSA嵌入的模块，通常用于处理辅助信息或补充特征（如外部MSA、辅助序列等）。它的结构和 MSA_emb 类似，但更简化，只输出 MSA 嵌入，不涉及 pair 或 state 嵌入。
    """
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_init=22+1+2, p_drop=0.1, input_seq_onehot=False):
        super(Extra_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(22, d_msa) # embedding for query sequence
        self.drop = nn.Dropout(p_drop)

        self.input_seq_onehot=input_seq_onehot

        self.reset_parameter()

    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L, 22)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        N = msa.shape[1] # number of sequenes in MSA
        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding

        # Sergey's one hot trick
        seq = (seq @ self.emb_q.weight).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = msa + seq.expand(-1, N, -1, -1) # adding query embedding to MSA
        return self.drop(msa)

class TemplatePairStack(nn.Module):
    # process template pairwise features
    # use structure-biased attention
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.25):
        super(TemplatePairStack, self).__init__()
        self.n_block = n_block
        proc_s = [PairStr2Pair(d_pair=d_templ, n_head=n_head, d_hidden=d_hidden, p_drop=p_drop) for i in range(n_block)]
        self.block = nn.ModuleList(proc_s)
        self.norm = nn.LayerNorm(d_templ)
    def forward(self, templ, rbf_feat, use_checkpoint=False):
        B, T, L = templ.shape[:3]
        templ = templ.reshape(B*T, L, L, -1)

        for i_block in range(self.n_block):
            if use_checkpoint:
                templ = checkpoint.checkpoint(create_custom_forward(self.block[i_block]), templ, rbf_feat)
            else:
                templ = self.block[i_block](templ, rbf_feat)
        return self.norm(templ).reshape(B, T, L, L, -1)

class TemplateTorsionStack(nn.Module):
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.15):
        super(TemplateTorsionStack, self).__init__()
        self.n_block=n_block
        self.proj_pair = nn.Linear(d_templ+36, d_templ)
        proc_s = [AttentionWithBias(d_in=d_templ, d_bias=d_templ,
                                    n_head=n_head, d_hidden=d_hidden) for i in range(n_block)]
        self.row_attn = nn.ModuleList(proc_s)
        proc_s = [FeedForwardLayer(d_templ, 4, p_drop=p_drop) for i in range(n_block)]
        self.ff = nn.ModuleList(proc_s)
        self.norm = nn.LayerNorm(d_templ)

    def reset_parameter(self):
        self.proj_pair = init_lecun_normal(self.proj_pair)
        nn.init.zeros_(self.proj_pair.bias)

    def forward(self, tors, pair, rbf_feat, use_checkpoint=False):
        B, T, L = tors.shape[:3]
        tors = tors.reshape(B*T, L, -1)
        pair = pair.reshape(B*T, L, L, -1)
        pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair)

        for i_block in range(self.n_block):
            if use_checkpoint:
                tors = tors + checkpoint.checkpoint(create_custom_forward(self.row_attn[i_block]), tors, pair)
            else:
                tors = tors + self.row_attn[i_block](tors, pair)
            tors = tors + self.ff[i_block](tors)
        return self.norm(tors).reshape(B, T, L, -1)

class Templ_emb(nn.Module):
    # Get template embedding
    # Features are
    #   t2d:
    #   - 37 distogram bins + 6 orientations (43)
    #   - Mask (missing/unaligned) (1)
    #   t1d:
    #   - tiled AA sequence (20 standard aa + gap)
    #   - confidence (1)
    #   - contacting or note (1). NB this is added for diffusion model. Used only in complex training examples - 1 signifies that a residue in the non-diffused chain\
    #     i.e. the context, is in contact with the diffused chain.
    #
    #Added extra t1d dimension for contacting or not
    """
    Templ_emb (Template Embedding) 的核心任务是将外部结构信息（即“模板”）`编码`并`融入到当前正在设计的蛋白质的特征表示中`（注意Templ_emb完成了两件事：编码外部结构信息、以及将外部结构信息融入目标特征）。这些外部信息可以来自：
        * 真实的PDB文件：在进行“局部扩散”或“基序支架”等任务时，提供一个已知的结构作为参考。
        * 上一步的预测结果：这是 RFdiffusion 中最核心的应用，即自条件（Self-Conditioning）。模型在时间步 t 预测出一个去噪的结构 p(x_0|x_t)，然后将这个预测结果作为模板，输入给 Templ_emb 模块，以辅助下一步（t-1）的预测。
    """
    def __init__(self, d_t1d=21+1+1, d_t2d=43+1, d_tor=30, d_pair=128, d_state=32,
                 n_block=2, d_templ=64,
                 n_head=4, d_hidden=16, p_drop=0.25):
        super(Templ_emb, self).__init__()

        # 对从模板结构中提取和拼接而成的2D特征图进行初步的嵌入处理
        self.emb = nn.Linear(d_t1d*2+d_t2d, d_templ)

        # 处理模板2D特征（“模板2D特征” 指的是从外部结构信息即“模板”中提取的2D几何特征，而不是目标序列的 pair 矩阵。）的小型 Evoformer 网络。它在模板内部进行信息交换，提炼出更具信息量的模板特征，然后再传递给主网络。
        self.templ_stack = TemplatePairStack(n_block=n_block, d_templ=d_templ, n_head=n_head,
                                             d_hidden=d_hidden, p_drop=p_drop)

        # 实现从模板到目标 pair 特征的信息注入。它使用当前的 pair 特征作为查询（Query），去查询经过 templ_stack 提炼后的模板特征（作为键/值 Key/Value）。注意力机制会计算出模板中哪些部分对当前的残基对最重要，并将这些信息加到 pair 特征上。
        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_pair)

        # 处理模板1D特征，包括序列信息和二面角（alpha_t）
        self.emb_t1d = nn.Linear(d_t1d+d_tor, d_templ)
        self.proj_t1d = nn.Linear(d_templ, d_templ)

        #self.tor_stack = TemplateTorsionStack(n_block=n_block, d_templ=d_templ, n_head=n_head,
        #                                      d_hidden=d_hidden, p_drop=p_drop)
        # 实现从模板到目标 state 特征的信息注入。用当前的 state 特征作为查询（Query），去查询处理后的1D模板特征（作为键/值 Key/Value），并将模板的1D信息（如二面角）融入到 state 特征中。
        self.attn_tor = Attention(d_state, d_templ, n_head, d_hidden, d_state)

        self.reset_parameter()

    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

        nn.init.kaiming_normal_(self.emb_t1d.weight, nonlinearity='relu')
        nn.init.zeros_(self.emb_t1d.bias)

        self.proj_t1d = init_lecun_normal(self.proj_t1d)
        nn.init.zeros_(self.proj_t1d.bias)

    def forward(self, t1d, t2d, alpha_t, xyz_t, pair, state, use_checkpoint=False):
        """
        将模板特征融入到目标的 pair 和 state 表征中。

        Args:
          - t1d: 模板的1D特征 (B, T, L, d_t1d)。包含氨基酸类型、置信度等。
          - t2d: 模板的2D特征 (B, T, L, L, d_t2d)。包含距离和方向信息。
          - alpha_t: 模板的二面角 (B, T, L, 30)。
          - xyz_t: 模板的原子坐标 (B, T, L, 27, 3)。
          - pair: 目标的初始pair表征 (B, L, L, d_pair)，将作为Query。
          - state: 目标的初始state表征 (B, L, d_state)，将作为Query。
        
        Returns:
          - pair: 融合了模板2D信息后的pair表征 (B, L, L, d_pair)。
          - state: 融合了模板1D信息后的state表征 (B, L, d_state)。
        """
        # Input
        #   - t1d: 1D template info (B, T, L, 23)
        #   - t2d: 2D template info (B, T, L, L, 44)
        B, T, L, _ = t1d.shape

        # Prepare 2D template features
        left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
        right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)
        #
        templ = torch.cat((t2d, left, right), -1) # (B, T, L, L, 90)
        templ = self.emb(templ) # Template templures (B, T, L, L, d_templ)
        # process each template features
        xyz_t = xyz_t.reshape(B*T, L, -1, 3)
        rbf_feat = rbf(torch.cdist(xyz_t[:,:,1], xyz_t[:,:,1]))     # rbf_feat: (B*T, L, L, 36)
        templ = self.templ_stack(templ, rbf_feat, use_checkpoint=use_checkpoint) # (B, T, L,L, d_templ)

        # Prepare 1D template torsion angle features
        t1d = torch.cat((t1d, alpha_t), dim=-1) # (B, T, L, 23+30)

        # process each template features
        t1d = self.proj_t1d(F.relu_(self.emb_t1d(t1d)))

        # mixing query state features to template state features
        state = state.reshape(B*L, 1, -1)
        t1d = t1d.permute(0,2,1,3).reshape(B*L, T, -1)
        if use_checkpoint:
            out = checkpoint.checkpoint(create_custom_forward(self.attn_tor), state, t1d, t1d)
            out = out.reshape(B, L, -1)
        else:
            out = self.attn_tor(state, t1d, t1d).reshape(B, L, -1)
        state = state.reshape(B, L, -1)
        state = state + out

        # mixing query pair features to template information (Template pointwise attention)
        pair = pair.reshape(B*L*L, 1, -1)
        templ = templ.permute(0, 2, 3, 1, 4).reshape(B*L*L, T, -1)
        if use_checkpoint:
            out = checkpoint.checkpoint(create_custom_forward(self.attn), pair, templ, templ)
            out = out.reshape(B, L, L, -1)
        else:
            out = self.attn(pair, templ, templ).reshape(B, L, L, -1)
        #
        pair = pair.reshape(B, L, L, -1)
        pair = pair + out

        return pair, state

class Recycling(nn.Module):
    """
    将上一步（或上一个循环）模型的输出特征（msa, pair, state, xyz）“回收”并处理后，作为当前网络的输入增强，实现信息的循环利用。
    """
    def __init__(self, d_msa=256, d_pair=128, d_state=32):
        super(Recycling, self).__init__()
        self.proj_dist = nn.Linear(36+d_state*2, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_msa = nn.LayerNorm(d_msa)

        self.reset_parameter()

    def reset_parameter(self):
        self.proj_dist = init_lecun_normal(self.proj_dist)
        nn.init.zeros_(self.proj_dist.bias)

    def forward(self, seq, msa, pair, xyz, state, attention_mask=None):
        B, L = pair.shape[:2]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1) # (B, L, 1)
            state = state * mask
            xyz = xyz * mask.unsqueeze(-1) # (B, L, 1, 1)
        
        state = self.norm_state(state)
        #
        left = state.unsqueeze(2).expand(-1,-1,L,-1)
        right = state.unsqueeze(1).expand(-1,L,-1,-1)

        # three anchor atoms
        N  = xyz[:,:,0]
        Ca = xyz[:,:,1]
        C  = xyz[:,:,2]

        # recreate Cb given N,Ca,C
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

        dist = rbf(torch.cdist(Cb, Cb))
        dist = torch.cat((dist, left, right), dim=-1)
        dist = self.proj_dist(dist)
        pair = dist + self.norm_pair(pair)
        msa = self.norm_msa(msa)
        return msa, pair, state

