import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import copy
import dgl
from model.rfdiffusion.util import base_indices, RTs_by_torsion, xyzs_in_base_frame, rigid_from_3_points


# def find_breaks(ix, thresh=35):
#     # finds positions in ix where the jump is greater than 100
#     breaks = np.where(np.diff(ix) > thresh)[0]
#     return np.array(breaks)+1


def init_lecun_normal(module):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape):
        stddev = np.sqrt(1.0/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    module.weight = torch.nn.Parameter( (sample_truncated_normal(module.weight.shape)) )
    return module

def init_lecun_normal_param(weight):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape):
        stddev = np.sqrt(1.0/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    weight = torch.nn.Parameter( (sample_truncated_normal(weight.shape)) )
    return weight

# for gradient checkpointing
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super(Dropout, self).__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-p_drop]))
        self.broadcast_dim=broadcast_dim
        self.p_drop=p_drop
    def forward(self, x):
        if not self.training: # no drophead during evaluation mode
            return x
        shape = list(x.shape)
        if not self.broadcast_dim == None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)

        x = mask * x / (1.0 - self.p_drop)
        return x

def rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 0., 20., 36
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

# def get_seqsep(idx, cyclic=None):
#     '''
#     Input:
#         - idx: residue indices of given sequence (B,L)
#     Output:
#         - seqsep: sequence separation feature with sign (B, L, L, 1)
#                   Sergey found that having sign in seqsep features helps a little
#     '''
#     seqsep = idx[:,None,:] - idx[:,:,None]
#     sign = torch.sign(seqsep)
#     neigh = torch.abs(seqsep)
#     neigh[neigh > 1] = 0.0 # if bonded -- 1.0 / else 0.0
#     neigh = sign * neigh

#     # add cyclic edges
#     breaks = find_breaks(idx.squeeze().cpu().numpy())
#     chainids = np.zeros_like(idx.squeeze().cpu().numpy())
#     for i, b in enumerate(breaks):
#         chainids[b:] = i+1
#     chainids = torch.from_numpy(chainids).to(device=idx.device)

#     # add cyclic edges with multiple chains
#     if (cyclic is not None):
#         for chid in torch.unique(chainids):
#             is_chid = chainids==chid
#             cur_cyclic = cyclic*is_chid
#             cur_cres = cur_cyclic.nonzero()

#             if cur_cyclic.sum()>=2:
#                 neigh[:,cur_cres[-1],cur_cres[0]] = 1
#                 neigh[:,cur_cres[0],cur_cres[-1]] = -1

#     return neigh.unsqueeze(-1)




def get_seqsep(idx, attention_mask, cyclic=None, thresh=35):
    '''
    生成序列分离特征的批量化、掩码安全且逻辑等价的版本。
    
    该函数执行以下操作：
    1. 计算残基邻接矩阵，逻辑与原始 get_seqsep 完全等价。
    2. 使用全向量化方法计算链ID，取代了旧的 find_breaks 和 for 循环，同时利用 attention_mask 确保了计算的正确性。
    3. 在批量级别上应用环化连接逻辑。

    Input:
        - idx (torch.Tensor): 形状为 (B, L) 的残基索引张量。可以包含填充值。
        - attention_mask (torch.Tensor): 形状为 (B, L) 的布尔掩码，True 代表有效残基。
        - cyclic (torch.Tensor, optional): 形状为 (B, L) 的布尔掩码，标记需要环化连接的残基。
        - thresh (int): 定义链断裂的索引差阈值。

    Output:
        - neigh (torch.Tensor): 形状为 (B, L, L, 1) 的序列分离特征张量。
    '''
    B, L = idx.shape
    device = idx.device

    # 1. 计算基础邻接矩阵 (完全批量化，逻辑与原始代码等价)
    #    seqsep: (B, L, L), 包含了所有残基对之间的索引差
    seqsep = idx[:, None, :] - idx[:, :, None]
    
    #    neigh: (B, L, L), 保留差值的绝对值为0或1的位置
    neigh = torch.abs(seqsep)
    neigh[neigh > 1] = 0.0  # 核心逻辑：这行代码完美复现了原始逻辑，保留了对角线和相邻对角线
    
    #    恢复符号
    neigh = torch.sign(seqsep) * neigh

    # 2. 高效且安全地计算 chain ID (完全向量化)
    #    只有在两个相邻残基都有效时，才计算它们之间的索引差
    #    首先，我们创建一个只包含有效残基索引的 "干净" 版本，填充区域用一个不会触发断裂的值（例如0）填充
    #    注意：这里的 `idx` 来自 collate_fn 的 `processed_idx`，padding value is -1
    #    所以 `idx * attention_mask.long()` 会将-1变为0，这在diff计算中可能会引入错误边界
    #    因此，更安全的方法是只在有效位置计算diff
    diffs = torch.diff(idx, dim=1)  # (B, L-1)
    
    #    创建一个掩码，仅当一对相邻残基都有效时才为 True
    valid_diff_mask = attention_mask[:, :-1] & attention_mask[:, 1:] # (B, L-1)

    #    仅在有效残基对之间检查索引差是否大于阈值
    breaks_matrix = (torch.abs(diffs) > thresh) & valid_diff_mask # (B, L-1), 布尔矩阵

    #    使用累加和 (cumsum) 来高效计算每个残基所属的链ID
    #    F.pad 在左侧填充0，使得 chain_ids 的长度恢复到 L
    chain_ids = F.pad(torch.cumsum(breaks_matrix.long(), dim=1), (1, 0), "constant", 0) # (B, L)

    # 3. 处理环化连接 (保留批次循环，因为内部逻辑复杂)
    if cyclic is not None:
        for i in range(B): # 遍历批次中的每个样本
            # 如果当前样本没有需要环化的残基，则跳过
            if not torch.any(cyclic[i]):
                continue

            cur_chain_ids = chain_ids[i]
            cur_cyclic = cyclic[i]
            
            # 找到所有出现过的链ID
            unique_chains = torch.unique(cur_chain_ids)

            for chain_id in unique_chains:
                # 找到当前链的所有残基
                in_chain_mask = (cur_chain_ids == chain_id)
                
                # 找到当前链中被标记为需要环化的残基的索引
                cyclic_res_in_chain = torch.where(cur_cyclic & in_chain_mask)[0]
                
                if cyclic_res_in_chain.numel() >= 2:
                    # 在该链的第一个和最后一个环化残基之间创建连接
                    # 注意：这里我们使用 .item() 来获取标量索引，以避免潜在的张量索引问题
                    start_res_idx = cyclic_res_in_chain[0].item()
                    end_res_idx = cyclic_res_in_chain[-1].item()
                    neigh[i, end_res_idx, start_res_idx] = 1
                    neigh[i, start_res_idx, end_res_idx] = -1

    return neigh.unsqueeze(-1)

def make_full_graph(xyz, pair, idx, top_k=64, kmin=9, attention_mask=None):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
        - attention_mask: (B, L) boolean mask, True for valid residues
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device
    
    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    
    # 初始条件：只要不是同一个残基就创建边
    cond = (sep.abs() > 0)

    # --- 新增的掩码逻辑 ---
    if attention_mask is not None:
        # 只在有效残基之间创建边
        pair_mask = attention_mask[:, :, None] & attention_mask[:, None, :]
        cond = cond & pair_mask
    # --- 结束新增 ---

    b, i, j = torch.where(cond)
   
    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['rel_pos'] = (xyz[b,j,:] - xyz[b,i,:]).detach() # no gradient through basis function

    return G, pair[b,i,j][...,None]

def make_topk_graph(xyz, pair, idx, top_k=64, kmin=32, eps=1e-6, attention_mask=None):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device
    
    # distance map from current CA coordinates
    D = torch.cdist(xyz, xyz) + torch.eye(L, device=device).unsqueeze(0)*999.9  # (B, L, L)
    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0)*999.9
    D = D + sep*eps
    
    # get top_k neighbors
    D_neigh, E_idx = torch.topk(D, min(top_k, L), largest=False) # shape of E_idx: (B, L, top_k)
    topk_matrix = torch.zeros((B, L, L), device=device)
    topk_matrix.scatter_(2, E_idx, 1.0)

    # put an edge if any of the 3 conditions are met:
    #   1) |i-j| <= kmin (connect sequentially adjacent residues)
    #   2) top_k neighbors
    cond = torch.logical_or(topk_matrix > 0.0, sep < kmin)

    # --- 修改开始 ---
    if attention_mask is not None:
        # 只在有效残基之间创建边
        pair_mask = attention_mask[:, :, None] & attention_mask[:, None, :]
        cond = cond & pair_mask
    # --- 修改结束 ---
    
    b,i,j = torch.where(cond)
   
    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['rel_pos'] = (xyz[b,j,:] - xyz[b,i,:]).detach() # no gradient through basis function

    return G, pair[b,i,j][...,None]

def make_rotX(angs, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    RTs[:,:,1,1] = angs[:,:,0]/NORM
    RTs[:,:,1,2] = -angs[:,:,1]/NORM
    RTs[:,:,2,1] = angs[:,:,1]/NORM
    RTs[:,:,2,2] = angs[:,:,0]/NORM
    return RTs

# rotate about the z axis
def make_rotZ(angs, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    RTs[:,:,0,0] = angs[:,:,0]/NORM
    RTs[:,:,0,1] = -angs[:,:,1]/NORM
    RTs[:,:,1,0] = angs[:,:,1]/NORM
    RTs[:,:,1,1] = angs[:,:,0]/NORM
    return RTs

# rotate about an arbitrary axis
def make_rot_axis(angs, u, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    ct = angs[:,:,0]/NORM
    st = angs[:,:,1]/NORM
    u0 = u[:,:,0]
    u1 = u[:,:,1]
    u2 = u[:,:,2]

    RTs[:,:,0,0] = ct+u0*u0*(1-ct)
    RTs[:,:,0,1] = u0*u1*(1-ct)-u2*st
    RTs[:,:,0,2] = u0*u2*(1-ct)+u1*st
    RTs[:,:,1,0] = u0*u1*(1-ct)+u2*st
    RTs[:,:,1,1] = ct+u1*u1*(1-ct)
    RTs[:,:,1,2] = u1*u2*(1-ct)-u0*st
    RTs[:,:,2,0] = u0*u2*(1-ct)-u1*st
    RTs[:,:,2,1] = u1*u2*(1-ct)+u0*st
    RTs[:,:,2,2] = ct+u2*u2*(1-ct)
    return RTs

class ComputeAllAtomCoords(nn.Module):
    def __init__(self):
        super(ComputeAllAtomCoords, self).__init__()

        self.base_indices = nn.Parameter(base_indices, requires_grad=False)
        self.RTs_in_base_frame = nn.Parameter(RTs_by_torsion, requires_grad=False)
        self.xyzs_in_base_frame = nn.Parameter(xyzs_in_base_frame, requires_grad=False)

    def forward(self, seq, xyz, alphas, non_ideal=False, use_H=True):
        B,L = xyz.shape[:2]

        Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], non_ideal=non_ideal)

        RTF0 = torch.eye(4).repeat(B,L,1,1).to(device=Rs.device)

        # bb
        RTF0[:,:,:3,:3] = Rs
        RTF0[:,:,:3,3] = Ts

        # omega
        RTF1 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq,0,:], make_rotX(alphas[:,:,0,:]))

        # phi
        RTF2 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,1,:], make_rotX(alphas[:,:,1,:]))

        # psi
        RTF3 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,2,:], make_rotX(alphas[:,:,2,:]))

        # CB bend
        basexyzs = self.xyzs_in_base_frame[seq]
        NCr = 0.5*(basexyzs[:,:,2,:3]+basexyzs[:,:,0,:3])
        CAr = (basexyzs[:,:,1,:3])
        CBr = (basexyzs[:,:,4,:3])
        CBrotaxis1 = (CBr-CAr).cross(NCr-CAr)
        CBrotaxis1 /= torch.linalg.norm(CBrotaxis1, dim=-1, keepdim=True)+1e-8
        
        # CB twist
        NCp = basexyzs[:,:,2,:3] - basexyzs[:,:,0,:3]
        NCpp = NCp - torch.sum(NCp*NCr, dim=-1, keepdim=True)/ torch.sum(NCr*NCr, dim=-1, keepdim=True) * NCr
        CBrotaxis2 = (CBr-CAr).cross(NCpp)
        CBrotaxis2 /= torch.linalg.norm(CBrotaxis2, dim=-1, keepdim=True)+1e-8
        
        CBrot1 = make_rot_axis(alphas[:,:,7,:], CBrotaxis1 )
        CBrot2 = make_rot_axis(alphas[:,:,8,:], CBrotaxis2 )
        
        RTF8 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, CBrot1,CBrot2)
        
        # chi1 + CG bend
        RTF4 = torch.einsum(
            'brij,brjk,brkl,brlm->brim', 
            RTF8, 
            self.RTs_in_base_frame[seq,3,:], 
            make_rotX(alphas[:,:,3,:]), 
            make_rotZ(alphas[:,:,9,:]))

        # chi2
        RTF5 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF4, self.RTs_in_base_frame[seq,4,:],make_rotX(alphas[:,:,4,:]))

        # chi3
        RTF6 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF5,self.RTs_in_base_frame[seq,5,:],make_rotX(alphas[:,:,5,:]))

        # chi4
        RTF7 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF6,self.RTs_in_base_frame[seq,6,:],make_rotX(alphas[:,:,6,:]))

        RTframes = torch.stack((
            RTF0,RTF1,RTF2,RTF3,RTF4,RTF5,RTF6,RTF7,RTF8
        ),dim=2)

        xyzs = torch.einsum(
            'brtij,brtj->brti', 
            RTframes.gather(2,self.base_indices[seq][...,None,None].repeat(1,1,1,4,4)), basexyzs
        )

        if use_H:
            return RTframes, xyzs[...,:3]
        else:
            return RTframes, xyzs[...,:14,:3]
