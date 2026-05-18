import torch
import torch.nn as nn
from model.rfdiffusion.Embeddings import MSA_emb, Extra_emb, Templ_emb, Recycling
from model.rfdiffusion.Track_module import IterativeSimulator
from model.rfdiffusion.AuxiliaryPredictor import DistanceNetwork, MaskedTokenNetwork, ExpResolvedNetwork, LDDTNetwork
from opt_einsum import contract as einsum

class RoseTTAFoldModule(nn.Module):
    """
    RoseTTAFoldModule 的任务是接收一个在时间步 t 的、带有噪声的蛋白质结构表示，并预测出它在 t=0 时的、无噪声的“干净”结构
    """
    def __init__(self, 
                 n_extra_block,              # MSA 堆栈中的额外 Evoformer 块数
                 n_main_block,               # 主干网络中的 Evoformer 块数
                 n_ref_block,                # 结构精修网络中的 Evoformer 块数
                 d_msa,                      # MSA 表征的维度
                 d_msa_full,                 # 全 MSA 表征的维度
                 d_pair,                     # 配对 (Pair) 表征的维度
                 d_templ,                    # 模板 (Template) 表征的维度
                 n_head_msa,                 # MSA 注意力头的数量
                 n_head_pair,                # 配对注意力头的数量
                 n_head_templ,               # 模板注意力头的数量
                 d_hidden,                   # Evoformer 内部前馈网络的隐藏层维度
                 d_hidden_templ,             # 模板网络内部前馈网络的隐藏层维度
                 p_drop,                     # Dropout 概率
                 d_t1d,                      # 1D 模板特征的维度
                 d_t2d,                      # 2D 模板特征的维度
                 T,                          # 扩散总步数 (用于时间步嵌入)
                 use_motif_timestep,         # 是否为 Motif 使用独立的时间步嵌入
                 freeze_track_motif,         # 是否在训练时冻结 Motif 追踪模块的参数
                 SE3_param_full={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, # 全原子 SE(3) Transformer 参数
                 SE3_param_topk={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, # 局部原子 SE(3) Transformer 参数
                 input_seq_onehot=False,     # 输入序列是否为 one-hot 编码
                 ):

        super(RoseTTAFoldModule, self).__init__()

        self.freeze_track_motif = freeze_track_motif
        
        # Input Embeddings
        d_state = SE3_param_topk['l0_out_features']
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, d_state=d_state,
                p_drop=p_drop, input_seq_onehot=input_seq_onehot) # Allowed to take onehotseq
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=25,
                p_drop=p_drop, input_seq_onehot=input_seq_onehot) # Allowed to take onehotseq
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, d_state=d_state,
                                   n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25, d_t1d=d_t1d, d_t2d=d_t2d)


        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        #
        self.simulator = IterativeSimulator(n_extra_block=n_extra_block,
                                            n_main_block=n_main_block,
                                            n_ref_block=n_ref_block,
                                            d_msa=d_msa, d_msa_full=d_msa_full,
                                            d_pair=d_pair, d_hidden=d_hidden,
                                            n_head_msa=n_head_msa,
                                            n_head_pair=n_head_pair,
                                            SE3_param_full=SE3_param_full,
                                            SE3_param_topk=SE3_param_topk,
                                            p_drop=p_drop)
        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa)
        self.lddt_pred = LDDTNetwork(d_state)
       
        self.exp_pred = ExpResolvedNetwork(d_msa, d_state)

    def forward(self, msa_latent, msa_full, seq, xyz, idx, t,
                t1d=None, t2d=None, xyz_t=None, alpha_t=None,
                msa_prev=None, pair_prev=None, state_prev=None,
                return_raw=False, return_full=False, return_infer=False,
                use_checkpoint=False, motif_mask=None, i_cycle=None, n_cycle=None,
                cyclic_reses=None, attention_mask=None):
        """
        Args:
            msa_latent: 潜空间 MSA 表征。形状: (B, N, L, d_msa)
            msa_full:   完整 MSA 表征。形状: (B, N_full, L, d_msa_full)
            seq:        输入序列的 one-hot 编码。形状: (B, L, 22)
            xyz:        输入的噪声原子坐标 (最多14个重原子)。形状: (B, L, 27, 3)
            idx:        残基索引。形状: (B, L)
            t:          当前的时间步。形状: (B,)
            t1d:        1D 模板特征。形状: (B, T, L, d_t1d)
            t2d:        2D 模板特征 (距离和角度)。形状: (B, T, L, L, d_t2d)
            xyz_t:      模板的原子坐标。形状: (B, T, L, 27, 3)
            alpha_t:    模板的二面角。形状: (B, T, L, 30)
            msa_prev:   上一步的 MSA 表征 (用于 recycling)。形状: (B, L, d_msa)
            pair_prev:  上一步的 Pair 表征 (用于 recycling)。形状: (B, L, L, d_pair)
            state_prev: 上一步的 State 表征 (用于 recycling)。形状: (B, L, d_state)
            motif_mask: Motif 区域的掩码，标记哪些残基是固定的。形状: (B, L)
            attention_mask: (B, L) 布尔张量，标记哪些是有效残基 (True) vs 填充残基 (False)

        Returns (for inference, when return_infer=True):
            msa:        输出的 MSA 表征。形状: (B, L, d_msa)
            pair:       输出的 Pair 表征。形状: (B, L, L, d_pair)
            xyz:        预测的去噪后原子坐标 (N, Cα, C)。形状: (B, L, 3, 3)，注意RoseTTAFoldModule模块仅预测和返回主链结构
            state:      输出的 State 表征。形状: (B, L, d_state)
            alpha:      预测的二面角。形状: (B, L, 7, 2)
            logits_aa:  预测的氨基酸序列 logits。形状: (B, L, 21)
            plddt:      预测的 pLDDT 值。形状: (B, L)
        """
        print("DEBUG-3")
        B, N, L = msa_latent.shape[:3]
        # Get embeddings
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx, cyclic_reses)
        msa_full = self.full_emb(msa_full, seq, idx)
        print("DEBUG-4")
        # Do recycling
        if msa_prev == None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
            pair_prev = torch.zeros_like(pair)
            state_prev = torch.zeros_like(state)
        msa_recycle, pair_recycle, state_recycle = self.recycle(seq, msa_prev, pair_prev, xyz, state_prev, attention_mask=attention_mask)
        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        state = state + state_recycle

        print("DEBUG-5")
        # Get timestep embedding (if using)
        if hasattr(self, 'timestep_embedder'):
            assert t is not None
            time_emb = self.timestep_embedder(L,t,motif_mask)
            n_tmpl = t1d.shape[1]
            t1d = torch.cat([t1d, time_emb[None,None,...].repeat(1,n_tmpl,1,1)], dim=-1)

        # add template embedding
        # 所有带有 T 维度的输入（t1d, t2d, xyz_t, alpha_t）都被送入了 self.templ_emb 模块。
        pair, state = self.templ_emb(t1d, t2d, alpha_t, xyz_t, pair, state, use_checkpoint=use_checkpoint)
        print("DEBUG-a")
        # Predict coordinates from given inputs
        is_frozen_residue = motif_mask if self.freeze_track_motif else torch.zeros_like(motif_mask).bool()
        msa, pair, R, T, alpha_s, state = self.simulator(seq, msa_latent, msa_full, pair, xyz[:,:,:3],
                                                         state, idx, use_checkpoint=use_checkpoint,
                                                         motif_mask=is_frozen_residue, cyclic_reses=cyclic_reses,
                                                         attention_mask=attention_mask)
        print("DEBUG-6")
        if return_raw:
            # get last structure
            # xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)：定义局部刚体，表示每个残基的 N, Cα, C 原子相对于其自身 Cα 原点的相对位置向量。xyz[:,:,:3]-xyz[:,:,1]是恒定不变的，变得是旋转矩阵R和平移矩阵T，模型学习到R和T，再与xyz[:,:,:3]-xyz[:,:,1]结合，得到新的绝对坐标
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            return msa[:,0], pair, xyz, state, alpha_s[-1]

        # predict masked amino acids
        # 接收最终的 MSA 特征 msa，并为每个位置预测一个包含21个值的向量（logits），分别对应20种标准氨基酸和1种未知/gap类型。
        logits_aa = self.aa_pred(msa)
        
        # Predict LDDT
        # 接收最终的 state 特征，并为每个残基预测其结构预测的准确性。输出的 lddt 通常是一个分布（logits over bins），而不是一个单一的标量值。
        lddt = self.lddt_pred(state)

        if return_infer:    # 专门用于推理（inference） 阶段
            # get last structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            
            # get scalar plddt
            nbin = lddt.shape[1]
            bin_step = 1.0 / nbin
            lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=lddt.dtype, device=lddt.device)
            pred_lddt = nn.Softmax(dim=1)(lddt)
            pred_lddt = torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

            return msa[:,0], pair, xyz, state, alpha_s[-1], logits_aa.permute(0,2,1), pred_lddt
        print("DEBUG-7")
        #
        # predict distogram & orientograms
        # 接收最终的 pair 特征，并预测残基对之间的6D坐标信息（1个距离，2个平面角，3个二面角），这是训练中的一个主要损失项。
        logits = self.c6d_pred(pair)
        
        # predict experimentally resolved or not
        # 预测每个残基是否在实验中可被解析，这是一个辅助的训练目标。
        logits_exp = self.exp_pred(msa[:,0], state)
        
        # get all intermediate bb structures
        xyz = einsum('rbnij,bnaj->rbnai', R, xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T.unsqueeze(-2)

        return logits, logits_aa, logits_exp, xyz, alpha_s, lddt
