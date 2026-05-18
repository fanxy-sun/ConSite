import torch.utils.checkpoint as checkpoint
from model.rfdiffusion.util_module import *
from model.rfdiffusion.Attention_module import *
from model.rfdiffusion.SE3_network import SE3TransformerWrapper

# Components for three-track blocks
# 1. MSA -> MSA update (biased attention. bias from pair & structure)
# 2. Pair -> Pair update (biased attention. bias from structure)
# 3. MSA -> Pair update (extract coevolution signal)
# 4. Str -> Str update (node from MSA, edge from Pair)

# Update MSA with biased self-attention. bias from Pair & Str
class MSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16,
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super(MSAPairStr2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.proj_pair = nn.Linear(d_pair+36, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        else:
            self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        
        # Do proper initialization
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distrib
        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_state = init_lecun_normal(self.proj_state)

        # initialize bias to zeros
        nn.init.zeros_(self.proj_pair.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, rbf_feat, state, attention_mask=None):
        '''
        Inputs:
            - msa: MSA feature (B, N, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
            - rbf_feat: Ca-Ca distance feature calculated from xyz coordinates (B, L, L, 36)
            - xyz: xyz coordinates (B, L, n_atom, 3)
            - state: updated node features after SE(3)-Transformer layer (B, L, d_state)
        Output:
            - msa: Updated MSA feature (B, N, L, d_msa)
        '''
        B, N, L = msa.shape[:3]

        # prepare input bias feature by combining pair & coordinate info
        pair = self.norm_pair(pair)
        pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair) # (B, L, L, d_pair)
        #
        # update query sequence feature (first sequence in the MSA) with feedbacks (state) from SE3
        state = self.norm_state(state)
        state = self.proj_state(state).reshape(B, 1, L, -1)
        msa = msa.index_add(1, torch.tensor([0,], device=state.device), state)
        #
        # Apply row/column attention to msa & transform 
        msa = msa + self.drop_row(self.row_attn(msa, pair, attention_mask=attention_mask))
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)

        return msa

class PairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        super(PairStr2Pair, self).__init__()
        
        self.emb_rbf = nn.Linear(d_rbf, d_hidden)
        self.proj_rbf = nn.Linear(d_hidden, d_pair)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_pair, 2)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.kaiming_normal_(self.emb_rbf.weight, nonlinearity='relu')
        nn.init.zeros_(self.emb_rbf.bias)
        
        self.proj_rbf = init_lecun_normal(self.proj_rbf)
        nn.init.zeros_(self.proj_rbf.bias)

    def forward(self, pair, rbf_feat, attention_mask=None):
        B, L = pair.shape[:2]

        rbf_feat = self.proj_rbf(F.relu_(self.emb_rbf(rbf_feat)))

        pair = pair + self.drop_row(self.row_attn(pair, rbf_feat, attention_mask=attention_mask))
        pair = pair + self.drop_col(self.col_attn(pair, rbf_feat, attention_mask=attention_mask))
        pair = pair + self.ff(pair)
        return pair

class MSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(MSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)
        
        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)

        # zero initialize output
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa, pair, attention_mask=None):
        B, N, L = msa.shape[:3]

        # --- 修改开始 ---
        # 如果提供了掩码，我们就在计算前主动将填充区域的特征清零。
        # 这确保了即使上游传入的填充值不是0，我们的计算也是正确的。
        if attention_mask is not None:
            # attention_mask: (B, L)
            # 我们需要将其扩展以匹配 msa 的形状 (B, N, L, D_msa)
            # unsqueeze(1) -> (B, 1, L)
            # unsqueeze(-1) -> (B, 1, L, 1)
            # .float() 将布尔掩码转为 0.0 和 1.0
            mask = attention_mask.unsqueeze(1).unsqueeze(-1).float()
            msa = msa * mask
        # --- 修改结束 ---

        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)
       
        pair = pair + out
        
        return pair

class SCPred(nn.Module):
    def __init__(self, d_msa=256, d_state=32, d_hidden=128, p_drop=0.15):
        super(SCPred, self).__init__()
        self.norm_s0 = nn.LayerNorm(d_msa)
        self.norm_si = nn.LayerNorm(d_state)
        self.linear_s0 = nn.Linear(d_msa, d_hidden)
        self.linear_si = nn.Linear(d_state, d_hidden)

        # ResNet layers
        self.linear_1 = nn.Linear(d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.linear_3 = nn.Linear(d_hidden, d_hidden)
        self.linear_4 = nn.Linear(d_hidden, d_hidden)

        # Final outputs
        self.linear_out = nn.Linear(d_hidden, 20)

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.linear_s0 = init_lecun_normal(self.linear_s0)
        self.linear_si = init_lecun_normal(self.linear_si)
        self.linear_out = init_lecun_normal(self.linear_out)
        nn.init.zeros_(self.linear_s0.bias)
        nn.init.zeros_(self.linear_si.bias)
        nn.init.zeros_(self.linear_out.bias)
        
        # right before relu activation: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_1.bias)
        nn.init.kaiming_normal_(self.linear_3.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_3.bias)

        # right before residual connection: zero initialize
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)
        nn.init.zeros_(self.linear_4.weight)
        nn.init.zeros_(self.linear_4.bias)
    
    def forward(self, seq, state):
        '''
        Predict side-chain torsion angles along with backbone torsions
        Inputs:
            - seq: hidden embeddings corresponding to query sequence (B, L, d_msa)
            - state: state feature (output l0 feature) from previous SE3 layer (B, L, d_state)
        Outputs:
            - si: predicted torsion angles (phi, psi, omega, chi1~4 with cos/sin, Cb bend, Cb twist, CG) (B, L, 10, 2)
        '''
        B, L = seq.shape[:2]
        seq = self.norm_s0(seq)
        state = self.norm_si(state)
        si = self.linear_s0(seq) + self.linear_si(state)

        si = si + self.linear_2(F.relu_(self.linear_1(F.relu_(si))))
        si = si + self.linear_4(F.relu_(self.linear_3(F.relu_(si))))

        si = self.linear_out(F.relu_(si))
        return si.view(B, L, 10, 2)


class Str2Str(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=16, 
            SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, p_drop=0.1):
        super(Str2Str, self).__init__()
        
        # initial node & pair feature process
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)
    
        self.embed_x = nn.Linear(d_msa+d_state, SE3_param['l0_in_features'])
        self.embed_e1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_e2 = nn.Linear(SE3_param['num_edge_features']+36+1, SE3_param['num_edge_features'])
        
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge1 = nn.LayerNorm(SE3_param['num_edge_features'])
        self.norm_edge2 = nn.LayerNorm(SE3_param['num_edge_features'])
        
        self.se3 = SE3TransformerWrapper(**SE3_param)
        self.sc_predictor = SCPred(d_msa=d_msa, d_state=SE3_param['l0_out_features'],
                                   p_drop=p_drop)
        
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_x = init_lecun_normal(self.embed_x)
        self.embed_e1 = init_lecun_normal(self.embed_e1)
        self.embed_e2 = init_lecun_normal(self.embed_e2)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_x.bias)
        nn.init.zeros_(self.embed_e1.bias)
        nn.init.zeros_(self.embed_e2.bias)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, cyclic_reses=None, top_k=64, eps=1e-5, attention_mask=None):
        B, N, L = msa.shape[:3]

        if motif_mask is None:
            motif_mask = torch.zeros(L).bool()
        
        # process msa & pair features
        node = self.norm_msa(msa[:,0])
        pair = self.norm_pair(pair)
        state = self.norm_state(state)
       
        node = torch.cat((node, state), dim=-1)
        node = self.norm_node(self.embed_x(node))
        pair = self.norm_edge1(self.embed_e1(pair))
        
        neighbor = get_seqsep(idx, attention_mask, cyclic_reses)
        rbf_feat = rbf(torch.cdist(xyz[:,:,1], xyz[:,:,1]))
        pair = torch.cat((pair, rbf_feat, neighbor), dim=-1)
        pair = self.norm_edge2(self.embed_e2(pair))
        
        # define graph
        if top_k != 0:
            G, edge_feats = make_topk_graph(xyz[:,:,1,:], pair, idx, top_k=top_k, attention_mask=attention_mask)
        else:
            G, edge_feats = make_full_graph(xyz[:,:,1,:], pair, idx, top_k=top_k, attention_mask=attention_mask)
        l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
        l1_feats = l1_feats.reshape(B*L, -1, 3)
        

        # apply SE(3) Transformer & update coordinates
        shift = self.se3(G, node.reshape(B*L, -1, 1), l1_feats, edge_feats)

        state = shift['0'].reshape(B, L, -1) # (B, L, C)
        
        offset = shift['1'].reshape(B, L, 2, 3)
        if motif_mask.dim() == 2:
            # 当 motif_mask 是 (B, L) 形状时，我们使用高级索引。
            # PyTorch 会将 (B, L) 的布尔掩码正确地应用到 (B, L, 2, 3) 的 offset 张量上，
            # 仅将掩码为 True 的对应 (b, l) 位置的 (2, 3) 子张量置零。
            # 这完美地实现了蛋白质独立操作原则。
            offset[motif_mask] = 0.0
        else: # motif_mask.dim() == 1
            # 为了保持向后兼容性或处理全局 motif 的情况，我们保留原始行为。
            # 这里 motif_mask 的形状是 (L,)。
            offset[:, motif_mask, ...] = 0.0

        delTi = offset[:,:,0,:] / 10.0 # translation
        R = offset[:,:,1,:] / 100.0 # rotation
        
        Qnorm = torch.sqrt( 1 + torch.sum(R*R, dim=-1) )
        qA, qB, qC, qD = 1/Qnorm, R[:,:,0]/Qnorm, R[:,:,1]/Qnorm, R[:,:,2]/Qnorm

        delRi = torch.zeros((B,L,3,3), device=xyz.device)
        delRi[:,:,0,0] = qA*qA+qB*qB-qC*qC-qD*qD
        delRi[:,:,0,1] = 2*qB*qC - 2*qA*qD
        delRi[:,:,0,2] = 2*qB*qD + 2*qA*qC
        delRi[:,:,1,0] = 2*qB*qC + 2*qA*qD
        delRi[:,:,1,1] = qA*qA-qB*qB+qC*qC-qD*qD
        delRi[:,:,1,2] = 2*qC*qD - 2*qA*qB
        delRi[:,:,2,0] = 2*qB*qD - 2*qA*qC
        delRi[:,:,2,1] = 2*qC*qD + 2*qA*qB
        delRi[:,:,2,2] = qA*qA-qB*qB-qC*qC+qD*qD

        Ri = einsum('bnij,bnjk->bnik', delRi, R_in)
        Ti = delTi + T_in #einsum('bnij,bnj->bni', delRi, T_in) + delTi
            
        alpha = self.sc_predictor(msa[:,0], state)
        return Ri, Ti, state, alpha

class IterBlock(nn.Module):
    """
    IterBlock主要功能是在三个核心信息轨道（MSA轨道、Pair轨道、Structure轨道）之间进行一次完整的信息交换和更新，从而共同优化蛋白质的序列、距离和三维结构表示。输入是 t 时刻的 MSA、Pair 和结构信息，输出是 t+1 时刻的、经过一轮优化的新信息。
    """
    def __init__(self, d_msa=256, d_pair=128,
                 n_head_msa=8, n_head_pair=4,
                 use_global_attn=False,
                 d_hidden=32, d_hidden_msa=None, p_drop=0.15,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        super(IterBlock, self).__init__()
        if d_hidden_msa == None:
            d_hidden_msa = d_hidden

        # MSA轨道内部更新。不仅考虑MSA内部的序列关系（通过行、列注意力），还接收来自Pair轨道和Structure轨道的信息作为偏置（bias）
        self.msa2msa = MSAPairStr2MSA(d_msa=d_msa, d_pair=d_pair,
                                      n_head=n_head_msa,
                                      d_state=SE3_param['l0_out_features'],
                                      use_global_attn=use_global_attn,
                                      d_hidden=d_hidden_msa, p_drop=p_drop)
        
        # 信息从 MSA轨道 -> Pair轨道。负责将从MSA中提炼出的共进化信号（coevolution signal）传递给Pair表征。
        self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair,
                                 d_hidden=d_hidden//2, p_drop=p_drop)
                                 #d_hidden=d_hidden, p_drop=p_drop)

        # Pair轨道内部更新。负责更新Pair表征。它使用三角注意力（axial attention）并接收来自Structure轨道的几何信息（rbf_feat）作为偏置
        self.pair2pair = PairStr2Pair(d_pair=d_pair, n_head=n_head_pair, 
                                      d_hidden=d_hidden, p_drop=p_drop)

        # Structure轨道更新。接收来自MSA轨道和Pair轨道的信息，通过SE(3) Transformer预测出对当前三维结构（由旋转矩阵R和平移向量T定义）的更新量，从而得到新的三维坐标和更新后的节点特征state
        self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair,
                               d_state=SE3_param['l0_out_features'],
                               SE3_param=SE3_param,
                               p_drop=p_drop)

    def forward(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, use_checkpoint=False, cyclic_reses=None, attention_mask=None):
        """
        Args:
            xyz: (B, L, 3, 3)。 注意这里xyz是绝对坐标不是相对坐标
        """
        rbf_feat = rbf(torch.cdist(xyz[:,:,1,:], xyz[:,:,1,:]))
        if use_checkpoint:
            msa = checkpoint.checkpoint(create_custom_forward(self.msa2msa), msa, pair, rbf_feat, state)
            pair = checkpoint.checkpoint(create_custom_forward(self.msa2pair), msa, pair)
            pair = checkpoint.checkpoint(create_custom_forward(self.pair2pair), pair, rbf_feat)
            R, T, state, alpha = checkpoint.checkpoint(create_custom_forward(self.str2str, top_k=0), msa, pair, R_in, T_in, xyz, state, idx, motif_mask, cyclic_reses)
        else:
            print("DEBUG-f")
            msa = self.msa2msa(msa, pair, rbf_feat, state, attention_mask=attention_mask)
            print("DEBUG-g")
            pair = self.msa2pair(msa, pair, attention_mask=attention_mask)
            print("DEBUG-e")
            pair = self.pair2pair(pair, rbf_feat, attention_mask=attention_mask)
            print("DEBUG-h")
            R, T, state, alpha = self.str2str(msa, pair, R_in, T_in, xyz, state, idx, motif_mask=motif_mask, cyclic_reses=cyclic_reses, top_k=0, attention_mask=attention_mask) 
            print("DEBUG-j")
        return msa, pair, R, T, state, alpha

class IterativeSimulator(nn.Module):
    """
    self.extra_block (由 n_extra_block 控制)
        含义: 宽泛的全局信息提取阶段。
        处理对象: msa_full，一个包含大量序列但特征维度较低（d_msa_full=64）的 MSA。
        核心机制: 使用带全局注意力 (use_global_attn=True) 的 IterBlock。全局注意力允许序列中的每个位置关注所有其他位置，非常适合从海量但低维的序列数据中快速捕捉长距离的共进化信号和蛋白质的整体特征。
        目标: 这个阶段的主要目标不是精细化 MSA 本身，而是利用 msa_full 的广度来快速、粗略地优化成对（pair）特征和结构（state）信息。
    self.main_block (由 n_main_block 控制)
        含义: 核心的深度信息提炼阶段。
        处理对象: msa，一个序列数量较少但特征维度更高（d_msa=256）的“种子”MSA (seed MSA)。
        核心机制: 使用标准的、非全局注意力 (use_global_attn=False) 的 IterBlock。
        目标: 对 msa、pair 和 state 进行深度迭代和联合优化，这是整个网络的主体计算部分
    self.str_refiner (由 n_ref_block 控制)
        含义: 最终的纯结构精修阶段。
        处理对象: 不再更新 MSA，而是直接利用 main_block 输出的 msa[:,0] (query sequence feature), pair 和 state 特征。
        核心机制: 直接调用 Str2Str 模块，这是一个专门的结构更新模块。它只执行“结构 -> 结构”的更新。值得注意的是，在这个阶段，SE(3) Transformer 的图构建方式也可能不同（例如，使用 top_k=64 的邻接图），意味着它更关注于局部几何的精修，而不是全局的更新。
        目标: 在 MSA 和 Pair 特征基本固定后，对蛋白质的 3D 坐标进行最后几轮的专门优化和微调，以获得更精确、更符合物理化学规律的结构。
    """
    def __init__(self, n_extra_block=4, n_main_block=12, n_ref_block=4,
                 d_msa=256, d_msa_full=64, d_pair=128, d_hidden=32,
                 n_head_msa=8, n_head_pair=4,
                 SE3_param_full={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 SE3_param_topk={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 p_drop=0.15):
        super(IterativeSimulator, self).__init__()
        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        self.n_ref_block = n_ref_block
        
        self.proj_state = nn.Linear(SE3_param_topk['l0_out_features'], SE3_param_full['l0_out_features'])
        # Update with extra sequences
        if n_extra_block > 0:
            self.extra_block = nn.ModuleList([IterBlock(d_msa=d_msa_full, d_pair=d_pair,
                                                        n_head_msa=n_head_msa,
                                                        n_head_pair=n_head_pair,
                                                        d_hidden_msa=8,
                                                        d_hidden=d_hidden,
                                                        p_drop=p_drop,
                                                        use_global_attn=True,
                                                        SE3_param=SE3_param_full)
                                                        for i in range(n_extra_block)])

        # Update with seed sequences
        if n_main_block > 0:
            self.main_block = nn.ModuleList([IterBlock(d_msa=d_msa, d_pair=d_pair,
                                                       n_head_msa=n_head_msa,
                                                       n_head_pair=n_head_pair,
                                                       d_hidden=d_hidden,
                                                       p_drop=p_drop,
                                                       use_global_attn=False,
                                                       SE3_param=SE3_param_full)
                                                       for i in range(n_main_block)])

        self.proj_state2 = nn.Linear(SE3_param_full['l0_out_features'], SE3_param_topk['l0_out_features'])
        # Final SE(3) refinement
        if n_ref_block > 0:
            self.str_refiner = Str2Str(d_msa=d_msa, d_pair=d_pair,
                                       d_state=SE3_param_topk['l0_out_features'],
                                       SE3_param=SE3_param_topk,
                                       p_drop=p_drop)
    
        self.reset_parameter()
    def reset_parameter(self):
        self.proj_state = init_lecun_normal(self.proj_state)
        nn.init.zeros_(self.proj_state.bias)
        self.proj_state2 = init_lecun_normal(self.proj_state2)
        nn.init.zeros_(self.proj_state2.bias)

    def forward(self, seq, msa, msa_full, pair, xyz_in, state, idx, cyclic_reses=None, use_checkpoint=False, motif_mask=None, attention_mask=None):
        """
        input:
           seq: query sequence (B, L)
           msa: seed MSA embeddings (B, N, L, d_msa)
           msa_full: extra MSA embeddings (B, N, L, d_msa_full)
           pair: initial residue pair embeddings (B, L, L, d_pair)
           xyz_in: initial BB coordinates (B, L, 3, 3)，注意传入self.simulator的xyz是经过切片的xyz[:,:,:3]，所以是(B, L, 3, 3)不是(B, L, 27, 3)
           state: initial state features containing mixture of query seq, sidechain, accuracy info (B, L, d_state)
           idx: residue index
           motif_mask: bool tensor, True if motif position that is frozen, else False(L,) 
        """

        B, L = pair.shape[:2]

        if motif_mask is None:
            motif_mask = torch.zeros(L).bool()

        # 旋转矩阵 (Rotation Matrix)。形状: (B, L, 3, 3)。为批量中每个残基都初始化一个单位旋转矩阵。R_in[b, i] 代表了第 b 个样本中第 i 个残基的局部坐标系的初始姿态（orientation）。在迭代过程中，SE(3) Transformer 会预测对这个姿态的更新delRi，通过矩阵乘法 delRi @ R_in得到新的旋转 Ri，从而旋转每个残基的骨架。
        R_in = torch.eye(3, device=xyz_in.device).reshape(1,1,3,3).expand(B, L, -1, -1)
        
        # 平移向量 (Translation Vector)。(B, L, 3)。xyz_in[:,:,1].clone() 提取了每个残基的 Cα 原子坐标作为其初始位置。T_in[b, i] 代表了第 b 个样本中第 i 个残基的局部坐标系的初始位置（position）。在迭代中，SE(3) Transformer 会预测对这个位置的更新（位移），从而移动每个残基。
        T_in = xyz_in[:,:,1].clone()
        
        # 局部相对坐标 (Local Relative Coordinates)。计算之后，xyz_in 不再是绝对坐标，而是每个残基骨架原子相对于其自身 Cα 原点的刚体坐标。`这个相对坐标系在迭代中是不变的`。整个迭代过程就是通过更新 R_in 和 T_in 来对这个刚体进行旋转和平移，从而得到新的绝对坐标 xyz = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)。
        xyz_in = xyz_in - T_in.unsqueeze(-2)
        
        state = self.proj_state(state)
        print("DEBUG-B")
        R_s = list()        # 存储每个迭代步骤的旋转矩阵 R_in
        T_s = list()        # 存储每个迭代步骤的平移向量 T_in
        alpha_s = list()    # 存储每个迭代步骤预测的氨基酸侧链扭转角 alpha
        for i_m in range(self.n_extra_block):
            R_in = R_in.detach() # detach rotation (for stability)
            T_in = T_in.detach()
            # Get current BB structure
            xyz = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)

            msa_full, pair, R_in, T_in, state, alpha = self.extra_block[i_m](msa_full, 
                                                                             pair,
                                                                             R_in, 
                                                                             T_in, 
                                                                             xyz, 
                                                                             state, 
                                                                             idx,
                                                                             motif_mask=motif_mask,
                                                                             use_checkpoint=use_checkpoint,
                                                                             cyclic_reses=cyclic_reses,
                                                                             attention_mask=attention_mask)
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)
        print("DEBUG-C")
        for i_m in range(self.n_main_block):
            R_in = R_in.detach()
            T_in = T_in.detach()
            # Get current BB structure
            xyz = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)
            
            msa, pair, R_in, T_in, state, alpha = self.main_block[i_m](msa, 
                                                                       pair,
                                                                       R_in, 
                                                                       T_in, 
                                                                       xyz, 
                                                                       state, 
                                                                       idx,
                                                                       motif_mask=motif_mask,
                                                                       use_checkpoint=use_checkpoint,
                                                                       cyclic_reses=cyclic_reses,
                                                                       attention_mask=attention_mask)
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)
        print("DEBUG-D")
        state = self.proj_state2(state)
        for i_m in range(self.n_ref_block):
            R_in = R_in.detach()
            T_in = T_in.detach()
            xyz = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)
            R_in, T_in, state, alpha = self.str_refiner(msa, 
                                                        pair, 
                                                        R_in, 
                                                        T_in, 
                                                        xyz, 
                                                        state, 
                                                        idx, 
                                                        top_k=64, 
                                                        motif_mask=motif_mask,
                                                        cyclic_reses=cyclic_reses,
                                                        attention_mask=attention_mask)
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

        R_s = torch.stack(R_s, dim=0)
        T_s = torch.stack(T_s, dim=0)
        alpha_s = torch.stack(alpha_s, dim=0)

        return msa, pair, R_s, T_s, alpha_s, state
