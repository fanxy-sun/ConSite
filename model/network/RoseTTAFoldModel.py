import torch
import torch.nn as nn
from model.network.Embeddings import MSA_emb, Pair_emb_wo_templ, Pair_emb_w_templ, Templ_emb
from model.network.Attention_module_w_str import IterativeFeatureExtractor
from model.network.DistancePredictor import DistanceNetwork
from model.network.Refine_module import Refine_module

class RoseTTAFold(nn.Module):
    """
    
    待优化：
        * 输入除主链原子外的其它原子，https://aistudio.google.com/prompts/1tCW388M0L5XtgozTJ0zYgRBirLUR-8GU
    """
    def __init__(self, n_module=4, n_module_str=4, n_layer=4,\
                 d_msa=64, d_pair=128, d_templ=64,\
                 n_head_msa=4, n_head_pair=8, n_head_templ=4,
                 d_hidden=64, r_ff=4, n_resblock=1, p_drop=0.1, 
                 performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, 
                 use_templ=False):
        super(RoseTTAFold, self).__init__()
        self.use_templ = use_templ
        #
        self.msa_emb = MSA_emb(d_model=d_msa, p_drop=p_drop, max_len=5000)
        if use_templ:
            self.templ_emb = Templ_emb(d_templ=d_templ, n_att_head=n_head_templ, r_ff=r_ff, 
                                       performer_opts=performer_L_opts, p_drop=0.0)
            self.pair_emb = Pair_emb_w_templ(d_model=d_pair, d_templ=d_templ, p_drop=p_drop)
        else:
            self.pair_emb = Pair_emb_wo_templ(d_model=d_pair, p_drop=p_drop)
        #
        self.feat_extractor = IterativeFeatureExtractor(n_module=n_module,\
                                                        n_module_str=n_module_str,\
                                                        n_layer=n_layer,\
                                                        d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden,\
                                                        n_head_msa=n_head_msa, \
                                                        n_head_pair=n_head_pair,\
                                                        r_ff=r_ff, \
                                                        n_resblock=n_resblock,
                                                        p_drop=p_drop,
                                                        performer_N_opts=performer_N_opts,
                                                        performer_L_opts=performer_L_opts,
                                                        SE3_param=SE3_param)
        # self.c6d_predictor = DistanceNetwork(d_pair, p_drop=p_drop)

    def forward(self, msa, seq, idx, xyz, seq_mask=None, msa_mask=None, t1d=None, t2d=None):
        """
        Args:
            msa (torch.Tensor): MSA序列, shape (B, N, L)
            seq (torch.Tensor): 目标序列, shape (B, L)
            idx (torch.Tensor): 残基索引, shape (B, L)
            xyz (torch.Tensor): 外部提供的原子坐标 (N, CA, C), shape (B, L, 3, 3)
            seq_mask (torch.Tensor, optional): 序列长度掩码, shape (B, L). Defaults to None.
            msa_mask (torch.Tensor, optional): MSA深度掩码, shape (B, N). Defaults to None.
            t1d (torch.Tensor, optional): 1D模板特征. Defaults to None.
            t2d (torch.Tensor, optional): 2D模板特征. Defaults to None.

        Returns:
            torch.Tensor: 1D特征 (msa_feat), shape (B, L, d_msa)
            torch.Tensor: 2D特征 (pair_feat), shape (B, L, L, d_pair)
        """
        B, N, L = msa.shape
        # Get embeddings
        msa_feat = self.msa_emb(msa, idx)   # msa_feat(B, N, L, d_msa)
        if self.use_templ:
            tmpl = self.templ_emb(t1d, t2d, idx, seq_mask)  # tmpl(B, L, L, d_templ)
            pair_feat = self.pair_emb(seq, idx, tmpl)       # pair_feat(B, L, L, d_pair)
        else:
            pair_feat = self.pair_emb(seq, idx)

        # seq_mask_1d: (B, L) -> (B, 1, L, 1) for msa_feat, (B, L, L, 1) for pair_feat
        # msa_mask_2d: (B, N) -> (B, N, 1, 1) for msa_feat
        seq_mask_msa = seq_mask.view(B, 1, L, 1)
        seq_mask_pair = (seq_mask.unsqueeze(1) * seq_mask.unsqueeze(2)).unsqueeze(-1)
        msa_mask_broad = msa_mask.view(B, N, 1, 1)

        msa_feat = msa_feat * msa_mask_broad * seq_mask_msa
        pair_feat = pair_feat * seq_mask_pair

        # Extract features
        seq1hot = torch.nn.functional.one_hot(seq, num_classes=21).float()  # (B, L, 21)
        msa_feat, pair_feat, state_feat = self.feat_extractor(msa_feat, pair_feat, seq1hot, idx, xyz, seq_mask, msa_mask)
        msa_feat_target = msa_feat[:, 0] * seq_mask.unsqueeze(-1)
        pair_feat = pair_feat * seq_mask_pair
        state_feat = state_feat * seq_mask.unsqueeze(-1)

        # msa_feat_target 是目标序列（第一条序列）的1D特征
        # pair_feat 是包含丰富空间信息的2D特征
        return msa_feat_target, pair_feat, state_feat

