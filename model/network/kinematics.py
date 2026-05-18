import numpy as np
import torch

PARAMS = {
    "DMIN"    : 2.0,
    "DMAX"    : 20.0,
    "DBINS"   : 36,
    "ABINS"   : 36,
}

# ============================================================
def get_pair_dist(a, b):
    """calculate pair distances between two sets of points
    
    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist

# ============================================================
def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c 

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v /= torch.norm(v, dim=-1, keepdim=True)
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v*w, dim=-1)

    return torch.acos(vw)

# ============================================================
def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)

    return torch.atan2(y, x)


# ============================================================
def xyz_to_c6d(xyz, seq_mask, params=PARAMS):
    """
    将笛卡尔坐标转换为c6d距离和方向图（优化版）。
    
    优化点:
    - 移除 torch.where 和稀疏索引赋值，改为完全向量化的操作。
    - 通过广播（broadcasting）为所有残基对准备计算输入。
    - 对所有残基对进行密集计算，然后使用最终掩码过滤掉无效结果。
    - 这种方法能更好地利用GPU并行计算能力，从而显著提升速度。

    Parameters
    ----------
    xyz : pytorch tensor of shape [batch,nres,3,3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    seq_mask: pytorch tensor of shape [batch, nres]
              A boolean mask where True indicates a real residue and False indicates padding.
    Returns
    -------
    c6d : pytorch tensor of shape [batch,nres,nres,4]
          stores stacked dist,omega,theta,phi 2D maps 
    """
    
    batch, nres = xyz.shape[:2]
    device = xyz.device

    # (修改点1: 提前创建 pair_mask)
    # 动机: 这是后续所有掩码操作的基础，用于区分真实残基对和填充对。
    pair_mask = seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1) # 使用逻辑与(&)更精确

    # three anchor atoms - (代码复用: 此部分逻辑不变)
    N  = xyz[:,:,0]
    Ca = xyz[:,:,1]
    C  = xyz[:,:,2]

    # recreate Cb given N,Ca,C - (代码复用: 此部分逻辑不变)
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # --- 距离计算 --- (代码复用: 此部分逻辑不变，但掩码应用方式改变)
    dist = get_pair_dist(Cb,Cb)
    dist[torch.isnan(dist)] = 999.9
    
    # (修改点2: 创建一个统一的计算掩码)
    # 动机: 将填充掩码和距离阈值掩码合并成一个掩码 `calc_mask`。
    # 只有当一个残基对 (i, j) 同时满足“都是真实残基”和“距离小于DMAX”时，才需要计算角度。
    dist_mask = dist < params['DMAX']
    calc_mask = pair_mask & dist_mask
    
    c6d = torch.zeros([batch,nres,nres,4],dtype=xyz.dtype,device=device)
    c6d[...,0] = dist + 999.9*torch.eye(nres,device=device)[None,...]
    
    # (修改点3: 向量化准备输入)
    # 动机: 为了能将整个 (B, L, L) 的残基对一次性送入角度计算函数，
    # 需要将原子坐标广播成 (B, L, L, 3) 的形状。
    # Ca_i, Cb_i 依赖于第一个残基索引 `i`
    Ca_i = Ca.unsqueeze(2).expand(-1,-1,nres,-1)
    Cb_i = Cb.unsqueeze(2).expand(-1,-1,nres,-1)
    N_i = N.unsqueeze(2).expand(-1,-1,nres,-1)
    
    # Ca_j, Cb_j 依赖于第二个残基索引 `j`
    Ca_j = Ca.unsqueeze(1).expand(-1,nres,-1,-1)
    Cb_j = Cb.unsqueeze(1).expand(-1,nres,-1,-1)

    # (修改点4: 向量化计算角度)
    # 动机: 在所有残基对上执行密集的角度计算。
    # get_dih 和 get_ang 内部的 PyTorch 操作本身就支持批处理，
    # 因此可以直接传入广播后的高维张量。
    dih_omega = get_dih(Ca_i, Cb_i, Cb_j, Ca_j)
    dih_theta = get_dih(N_i, Ca_i, Cb_i, Cb_j)
    ang_phi = get_ang(Ca_i, Cb_i, Cb_j)

    # (修改点5: 使用掩码进行密集赋值)
    # 动机: 用之前创建的 calc_mask 将计算出的角度值填充到 c6d 张量中。
    # `torch.where` 比稀疏索引赋值更高效。
    c6d[...,1] = torch.where(calc_mask, dih_omega, c6d[...,1])
    c6d[...,2] = torch.where(calc_mask, dih_theta, c6d[...,2])
    c6d[...,3] = torch.where(calc_mask, ang_phi, c6d[...,3])
    
    # fix long-range distances - (代码复用: 此部分逻辑不变)
    c6d[...,0][c6d[...,0]>=params['DMAX']] = 999.9
    
    # (修改点6: 简化 final_mask 的生成和最终清理)
    # 动机: 逻辑更清晰，直接返回有效的计算掩码，并确保填充区域为零。
    final_mask = calc_mask.float()
    c6d = c6d * pair_mask.unsqueeze(-1)

    return c6d, final_mask
    
def rosetta_xyz_to_t2d(xyz_t, t0d, seq_mask, params=PARAMS):
    """convert template cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz_t : pytorch tensor of shape [batch,templ,nres,3,3]
            stores Cartesian coordinates of template backbone N,Ca,C atoms
    t0d:  0-D template features (HHprob, seqID, similarity) [batch, templ, 3]

    Returns
    -------
    t2d : pytorch tensor of shape [batch,nres,nres,1+6+3]
          stores stacked dist,omega,theta,phi 2D maps 
    """
    B, T, L = xyz_t.shape[:3]
    seq_mask_T = seq_mask.unsqueeze(1).expand(-1, T, -1).reshape(B*T, L)
    c6d, mask = xyz_to_c6d(xyz_t.view(B*T,L,3,3), seq_mask=seq_mask_T, params=params)
    c6d = c6d.view(B, T, L, L, 4)
    mask = mask.view(B, T, L, L, 1)
    #
    dist = c6d[...,:1] / params['DMAX'] # from 0 to 1 # (B, T, L, L, 1)
    dist = torch.clamp(dist, 0.0, 1.0)
    orien = torch.cat((torch.sin(c6d[...,1:]), torch.cos(c6d[...,1:])), dim=-1) # (B, T, L, L, 6)
    dist = dist * mask
    orien = orien * mask
    t0d = t0d.unsqueeze(2).unsqueeze(3).expand(-1, -1, L, L, -1)
    #
    t2d = torch.cat((dist, orien, t0d), dim=-1)
    t2d[torch.isnan(t2d)] = 0.0
    return t2d

# ============================================================
def c6d_to_bins(c6d,params=PARAMS):
    """bin 2d distance and orientation maps
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    astep = 2.0*np.pi / params['ABINS']

    dbins = torch.linspace(params['DMIN']+dstep, params['DMAX'], params['DBINS'],dtype=c6d.dtype,device=c6d.device)
    ab360 = torch.linspace(-np.pi+astep, np.pi, params['ABINS'],dtype=c6d.dtype,device=c6d.device)
    ab180 = torch.linspace(astep, np.pi, params['ABINS']//2,dtype=c6d.dtype,device=c6d.device)

    db = torch.bucketize(c6d[...,0].contiguous(),dbins)
    ob = torch.bucketize(c6d[...,1].contiguous(),ab360)
    tb = torch.bucketize(c6d[...,2].contiguous(),ab360)
    pb = torch.bucketize(c6d[...,3].contiguous(),ab180)

    ob[db==params['DBINS']] = params['ABINS']
    tb[db==params['DBINS']] = params['ABINS']
    pb[db==params['DBINS']] = params['ABINS']//2

    return torch.stack([db,ob,tb,pb],axis=-1).to(torch.uint8)


# ============================================================
def dist_to_bins(dist,params=PARAMS):
    """bin 2d distance maps
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    db = torch.round((dist-params['DMIN']-dstep/2)/dstep)

    db[db<0] = 0
    db[db>params['DBINS']] = params['DBINS']
    
    return db.long()


# ============================================================
def c6d_to_bins2(c6d,params=PARAMS):
    """bin 2d distance and orientation maps
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    astep = 2.0*np.pi / params['ABINS']

    db = torch.round((c6d[...,0]-params['DMIN']-dstep/2)/dstep)
    ob = torch.round((c6d[...,1]+np.pi-astep/2)/astep)
    tb = torch.round((c6d[...,2]+np.pi-astep/2)/astep)
    pb = torch.round((c6d[...,3]-astep/2)/astep)

    # put all d<dmin into one bin
    db[db<0] = 0
    
    # synchronize no-contact bins
    db[db>params['DBINS']] = params['DBINS']
    ob[db==params['DBINS']] = params['ABINS']
    tb[db==params['DBINS']] = params['ABINS']
    pb[db==params['DBINS']] = params['ABINS']//2
    
    return torch.stack([db,ob,tb,pb],axis=-1).long()
