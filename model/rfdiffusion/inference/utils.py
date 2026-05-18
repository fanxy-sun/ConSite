import numpy as np
import os
from omegaconf import DictConfig
import torch
import torch.nn.functional as nn
from model.rfdiffusion.diffusion import get_beta_schedule
from scipy.spatial.transform import Rotation as scipy_R
from model.rfdiffusion.util import rigid_from_3_points
from model.rfdiffusion.util_module import ComputeAllAtomCoords
from model.rfdiffusion import util
import random
import logging
from model.rfdiffusion.inference import model_runners
import glob

###########################################################
#### Functions which can be called outside of Denoiser ####
###########################################################


def batch_get_next_frames(xt, px0, t, T, diffuser, so3_type, diffusion_mask, noise_scale=1.0):
    """
    get_next_frames 的批量化 GPU 版本（重新设计）。
    此版本明确处理了从 Gram-Schmidt 过程可能产生的瑕旋转矩阵（行列式为-1）的问题。
    它通过 SVD 在内部强制校正旋转矩阵，确保所有后续操作都在有效的 SO(3) 流形上进行，
    从而不再依赖于 rigid_from_3_points 函数的特定版本。

    Args:
        xt (torch.Tensor): 当前时间步 t 的噪声坐标，形状为 [L, 14, 3]，其中 L 是批次中所有蛋白质残基的总和。
        px0 (torch.Tensor): 模型预测的 t=0 时的坐标，形状为 [L, 14, 3]。
        t (int): 当前的离散时间步 (例如, 从 T-1 到 0)。
        T (int): 扩散过程的总时间步数。
        diffuser: 包含 SO(3) 扩散器和预计算值的 Diffuser 对象。
        so3_type (str): 使用的 SO(3) 扩散类型 (此处预期为 'igso3')。
        diffusion_mask (torch.Tensor):布尔张量，形状为 [L]。True 表示不更新该残基 (例如 motif)。
        noise_scale (float): 添加到更新中的噪声的缩放因子 (仅限 IGSO3)。

    Returns:
        torch.Tensor: 更新后的 t-1 时间步的 backbone 坐标 (N, C-alpha, C)，形状为 [L, 3, 3]。
    """
    if so3_type != "igso3":
        raise NotImplementedError(f"so3 diffusion type {so3_type} not implemented in batch mode")

    device = xt.device
    dtype = xt.dtype
    L = xt.shape[0]

    # 1. 从原子坐标计算刚体变换，不依赖 det 参数
    N_0, Ca_0, C_0 = px0[None, :, 0, :], px0[None, :, 1, :], px0[None, :, 2, :]
    R_0_raw, Ca_0 = rigid_from_3_points(N_0, Ca_0, C_0)

    N_t, Ca_t, C_t = xt[None, :, 0, :], xt[None, :, 1, :], xt[None, :, 2, :]
    R_t_raw, Ca_t = rigid_from_3_points(N_t, Ca_t, C_t)

    # 2. 【核心设计变更】使用 SVD 手动校正旋转矩阵，确保行列式为 +1
    # 这是为了替代 scipy.spatial.transform.Rotation 的内部校正功能
    def correct_rotation_matrix(R_raw):
        # R_raw 形状: (1, L, 3, 3)
        U, _, Vt = torch.linalg.svd(R_raw)
        # 确保为右手坐标系 (det(R) = +1)
        # det_UVt = det(U) * det(Vt)
        det_UVt = torch.det(U @ Vt)
        
        # 构造修正矩阵 D = diag(1, 1, det(UVt))
        correction = torch.eye(3, device=device, dtype=dtype).expand_as(R_raw).clone()
        correction[..., -1, -1] = det_UVt
        
        # R_corrected = U @ D @ Vt
        R_corrected = U @ correction @ Vt
        return R_corrected

    R_0 = correct_rotation_matrix(R_0_raw)
    R_t = correct_rotation_matrix(R_t_raw)

    # 3. 移除临时的批次维度，后续操作的张量形状与原函数中的 numpy 数组保持一致
    R_0, Ca_0 = R_0.squeeze(0), Ca_0.squeeze(0)
    R_t, Ca_t = R_t.squeeze(0), Ca_t.squeeze(0)

    # 4. 计算从 R_t 到 R_0 的相对旋转，并近似其对数映射（旋转向量）
    R_t0 = torch.einsum("lij,lkj->lik", R_t, R_0)
    
    trace = torch.einsum('lii->l', R_t0)
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
    sin_theta = torch.sin(theta)
    axis_unnormalized = torch.stack([
        R_t0[:, 2, 1] - R_t0[:, 1, 2],
        R_t0[:, 0, 2] - R_t0[:, 2, 0],
        R_t0[:, 1, 0] - R_t0[:, 0, 1]
    ], dim=-1)
    rotvec = torch.where(
        (sin_theta.abs() > 1e-6)[:, None],
        axis_unnormalized * (theta / (2 * sin_theta))[:, None],
        torch.zeros(L, 3, device=device, dtype=dtype)
    )
    
    # 5. 使用预计算表插值计算分数范数
    Omega = torch.linalg.norm(rotvec, dim=-1)
    score_norm_vals = torch.from_numpy(diffuser.so3_diffuser.igso3_vals["score_norm"][diffuser.so3_diffuser.t_to_idx(t)]).to(device, dtype)
    discrete_omega = torch.from_numpy(diffuser.so3_diffuser.igso3_vals["discrete_omega"]).to(device, dtype)
    indices = torch.searchsorted(discrete_omega, Omega)
    idx_left = (indices - 1).clamp(min=0)
    idx_right = indices.clamp(max=len(discrete_omega) - 1)
    omega_left, omega_right = discrete_omega[idx_left], discrete_omega[idx_right]
    score_left, score_right = score_norm_vals[idx_left], score_norm_vals[idx_right]
    weight = (Omega - omega_left) / (omega_right - omega_left).clamp(min=1e-9)
    score_norm_t = score_left + weight * (score_right - score_left)
    Score_approx = rotvec * (score_norm_t / Omega.clamp(min=1e-9)).unsqueeze(-1)
    Score_approx = torch.nan_to_num(Score_approx)

    # 6. 计算 SDE 更新项
    continuous_t = (t + 1) / T
    rot_g = diffuser.so3_diffuser.g(continuous_t).to(device, dtype)
    Z = torch.randn(L, 3, device=device, dtype=dtype) * noise_scale
    Delta_r = (rot_g**2) * diffuser.so3_diffuser.step_size * Score_approx
    Perturb_tangent = Delta_r + rot_g * np.sqrt(diffuser.so3_diffuser.step_size) * Z
    Perturb_tangent[diffusion_mask] = 0.0

    # 7. 将切空间向量转为旋转矩阵并应用更新
    all_rot_transitions = diffuser.so3_diffuser.rodrigues_formula(Perturb_tangent)
    all_rot_transitions = all_rot_transitions[:, None, :, :]
    centered_coords = xt[:, :3, :] - Ca_t[:, None, :]
    rotated_coords = torch.einsum("lrij,laj->lrai", all_rot_transitions, centered_coords)
    next_crds = rotated_coords + Ca_t[:, None, None, :]

    return next_crds.squeeze(1)


def get_mu_xt_x0(xt, px0, t, beta_schedule, alphabar_schedule, eps=1e-6):
    """
    计算DDPM后验分布 p(x_{t-1} | x_t, x_0) 的均值和方差
    """
    # sigma is predefined from beta. Often referred to as beta tilde t
    t_idx = t - 1
    sigma = (
        (1 - alphabar_schedule[t_idx - 1]) / (1 - alphabar_schedule[t_idx])
    ) * beta_schedule[t_idx]        # 方差

    xt_ca = xt[:, 1, :]     # (L, 3)
    px0_ca = px0[:, 1, :]   # (L, 3)

    a = (
        (torch.sqrt(alphabar_schedule[t_idx - 1] + eps) * beta_schedule[t_idx])
        / (1 - alphabar_schedule[t_idx])
    ) * px0_ca
    b = (
        (
            torch.sqrt(1 - beta_schedule[t_idx] + eps)
            * (1 - alphabar_schedule[t_idx - 1])
        )
        / (1 - alphabar_schedule[t_idx])
    ) * xt_ca

    mu = a + b  # 均值

    return mu, sigma


def get_next_ca(
    xt,
    px0,
    t,
    diffusion_mask,
    crd_scale,
    beta_schedule,
    alphabar_schedule,
    noise_scale=1.0,
):
    """
    在扩散模型的反向过程（去噪）中，get_next_ca 函数负责执行单步平移去噪。给定当前时间步 t 的噪声结构 xt 和神经网络对最终“干净”结构 x0 的预测 px0，该函数的任务是计算并返回上一个时间步 t-1 的结构 x_{t-1}。这个过程只针对蛋白质骨架的**位置（平移）**部分，具体体现在它只直接更新 C-alpha (CA) 原子的坐标，然后将这个更新平移地应用到整个残基骨架上。旋转部分的去噪由另一个函数（如 get_next_frames）处理。

    Parameters:
        xt (L, 14/27, 3) set of coordinates
        px0 (L, 14/27, 3) set of coordinates
        t: time step. 1-based index
        logits_aa (L x 20 ) amino acid probabilities at each position
        seq_schedule (L): Tensor of bools, True is unmasked, False is masked. For this specific t
        diffusion_mask (torch.tensor, required): Tensor of bools, True means NOT diffused at this residue, False means diffused
        noise_scale: scale factor for the noise being added

    """
    # bring to origin after global alignment (when don't have a motif) or replace input motif and bring to origin, and then scale
    px0 = px0 * crd_scale
    xt = xt * crd_scale

    # get mu(xt, x0)
    mu, sigma = get_mu_xt_x0(
        xt, px0, t, beta_schedule=beta_schedule, alphabar_schedule=alphabar_schedule
    )

    sampled_crds = torch.normal(mu, torch.sqrt(sigma * noise_scale))
    delta = sampled_crds - xt[:, 1, :]  # check sign of this is correct

    if not diffusion_mask is None:
        # Don't move motif
        delta[diffusion_mask, ...] = 0

    out_crds = xt + delta[:, None, :]

    return out_crds / crd_scale, delta / crd_scale


def get_noise_schedule(T, noiseT, noise1, schedule_type):
    """
    Function to create a schedule that varies the scale of noise given to the model over time

    Parameters:

        T: The total number of timesteps in the denoising trajectory

        noiseT: The inital (t=T) noise scale

        noise1: The final (t=1) noise scale

        schedule_type: The type of function to use to interpolate between noiseT and noise1

    Returns:

        noise_schedule: A function which maps timestep to noise scale

    """

    noise_schedules = {
        "constant": lambda t: noiseT,
        "linear": lambda t: ((t - 1) / (T - 1)) * (noiseT - noise1) + noise1,
    }

    assert (
        schedule_type in noise_schedules
    ), f"noise_schedule must be one of {noise_schedules.keys()}. Received noise_schedule={schedule_type}. Exiting."

    return noise_schedules[schedule_type]


class Denoise:
    """
    Class for getting x(t-1) from predicted x0 and x(t)
    Strategy:
        Ca coordinates: Rediffuse to x(t-1) from predicted x0
        Frames: Approximate update from rotation score
        Torsions: 1/t of the way to the x0 prediction

    """

    def __init__(
        self,
        T,
        L,
        diffuser,
        b_0=0.001,
        b_T=0.1,
        min_b=1.0,
        max_b=12.5,
        min_sigma=0.05,
        max_sigma=1.5,
        noise_level=0.5,
        schedule_type="linear",
        so3_schedule_type="linear",
        schedule_kwargs={},
        so3_type="igso3",
        noise_scale_ca=1.0,
        final_noise_scale_ca=1,
        ca_noise_schedule_type="constant",
        noise_scale_frame=0.5,
        final_noise_scale_frame=0.5,
        frame_noise_schedule_type="constant",
        crd_scale=1 / 15,
        potential_manager=None,
        partial_T=None,
        device='cpu'
    ):
        """

        Parameters:
            noise_level: scaling on the noise added (set to 0 to use no noise,
                to 1 to have full noise)

        """
        self.T = T
        self.L = L
        self.diffuser = diffuser
        self.b_0 = b_0
        self.b_T = b_T
        self.noise_level = noise_level
        self.schedule_type = schedule_type
        self.so3_type = so3_type
        self.crd_scale = crd_scale
        self.noise_scale_ca = noise_scale_ca
        self.final_noise_scale_ca = final_noise_scale_ca
        self.ca_noise_schedule_type = ca_noise_schedule_type
        self.noise_scale_frame = noise_scale_frame
        self.final_noise_scale_frame = final_noise_scale_frame
        self.frame_noise_schedule_type = frame_noise_schedule_type
        self.potential_manager = potential_manager
        self._log = logging.getLogger(__name__)
        self.device = device

        schedule, alpha_schedule, alphabar_schedule = get_beta_schedule(
            self.T, self.b_0, self.b_T, self.schedule_type, inference=True
        )
        self.schedule = schedule.to(device)
        self.alpha_schedule = alpha_schedule.to(device)
        self.alphabar_schedule = alphabar_schedule.to(device)

        self.noise_schedule_ca = get_noise_schedule(
            self.T,
            self.noise_scale_ca,
            self.final_noise_scale_ca,
            self.ca_noise_schedule_type,
        )
        self.noise_schedule_frame = get_noise_schedule(
            self.T,
            self.noise_scale_frame,
            self.final_noise_scale_frame,
            self.frame_noise_schedule_type,
        )

    @property
    def idx2steps(self):
        return self.decode_scheduler.idx2steps.numpy()

    def batch_align_to_xt_motif(self, px0: torch.Tensor, xt: torch.Tensor, diffusion_mask: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-6):
        """
        align_to_xt_motif 的批量化 GPU 版本。
        使用 Kabsch 算法将 px0 中的每个蛋白质的 motif 对齐到 xt 中对应蛋白质的 motif。
        $$ atom_mask可能有问题
        
        Args:
            px0 (torch.Tensor): 去噪模型根据 xt 预测出的在 t=0 时的“干净”结构。 (N_total, n_atom, 3)。
            xt (torch.Tensor): 当前时间步 t 的噪声原子坐标 (N_total, n_atom, 3)。
            diffusion_mask (torch.Tensor): Motif 掩码 (N_total,)。
            lengths (torch.Tensor): 每个样本的长度 (B,)。
        
        Returns:
            torch.Tensor: 对齐后的 px0，形状与输入 px0 相同。
        """
        device = px0.device
        dtype = px0.dtype
        batch_size = lengths.shape[0]
        N_total, n_atom, _ = px0.shape

        # 关键修正：处理NaN值，与原始函数行为保持一致
        atom_mask = ~torch.isnan(px0)
        px0_no_nan = torch.nan_to_num(px0, nan=0.0)

        # 创建一个索引，将每个残基映射到其批次中的样本索引
        batch_idx = torch.repeat_interleave(torch.arange(batch_size, device=device), lengths)

        # 1. 提取并中心化 Motif 坐标 (按蛋白质样本分组)
        if diffusion_mask.sum() == 0:
             # 如果没有任何 motif 残基，直接返回原始 px0
            return px0

        # 提取 motif 的 BB 原子坐标，并展平为点云
        px0_motif_atoms = px0_no_nan[diffusion_mask, :3].reshape(-1, 3)
        xt_motif_atoms = xt[diffusion_mask, :3].reshape(-1, 3)
        
        # 关键修正：创建原子级别的批次索引
        motif_res_batch_idx = batch_idx[diffusion_mask]
        motif_atom_batch_idx = motif_res_batch_idx.repeat_interleave(3)

        # 计算每个样本 motif 的质心
        px0_motif_mean = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        xt_motif_mean = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        
        px0_motif_mean.index_add_(0, motif_atom_batch_idx, px0_motif_atoms)
        xt_motif_mean.index_add_(0, motif_atom_batch_idx, xt_motif_atoms)
        
        # 关键修正：使用正确的原子数计算平均值
        num_motif_atoms_per_protein = torch.bincount(motif_atom_batch_idx, minlength=batch_size).unsqueeze(1).clamp(min=1)
        px0_motif_mean /= num_motif_atoms_per_protein
        xt_motif_mean /= num_motif_atoms_per_protein

        # 中心化 motif 坐标
        px0_motif_centered = px0_motif_atoms - px0_motif_mean[motif_atom_batch_idx]
        xt_motif_centered = xt_motif_atoms - xt_motif_mean[motif_atom_batch_idx]

        # 2. 批量化 Kabsch 算法
        A = xt_motif_centered
        B = px0_motif_centered

        # 计算协方差矩阵 C = A^T * B (按样本分组)
        C = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        # 使用 einsum 计算每个原子对的外积，然后用 index_add_ 按样本求和
        C.index_add_(0, motif_atom_batch_idx, torch.einsum('bi,bj->bij', A, B))
        
        # 批量 SVD
        U, S, Vt = torch.linalg.svd(C)

        # 确保为右手坐标系
        det = torch.det(Vt.transpose(-2, -1) @ U.transpose(-2, -1))
        d = torch.eye(3, device=device, dtype=dtype).expand(batch_size, -1, -1).clone()
        d[:, -1, -1] = det

        # 计算旋转矩阵 R
        R = Vt.transpose(-2, -1) @ d @ U.transpose(-2, -1) # (B, 3, 3)

        # 3. 对整个 px0 应用旋转和平移
        # a. 将整个 px0 (所有原子) 展平并中心化 (使用各自的 motif 中心)
        px0_flat = px0_no_nan.reshape(-1, 3)
        # 关键修正：创建适用于所有原子的批次索引
        atom_batch_idx = batch_idx.repeat_interleave(n_atom)
        px0_flat_centered = px0_flat - px0_motif_mean[atom_batch_idx]

        # b. 应用旋转
        R_broadcasted = R[atom_batch_idx] # (N_total * n_atom, 3, 3)
        px0_flat_rotated = torch.einsum('ni,nij->nj', px0_flat_centered, R_broadcasted)

        # c. 平移回目标位置
        px0_flat_aligned = px0_flat_rotated + xt_motif_mean[atom_batch_idx]
        
        # d. 恢复原始形状
        px0_aligned = px0_flat_aligned.reshape(N_total, n_atom, 3)

        # 关键修正：恢复原始的 NaN 值
        px0_aligned[~atom_mask] = float("nan")

        return px0_aligned

    def get_potential_gradients(self, xyz, diffusion_mask):
        """
        This could be moved into potential manager if desired - NRB

        Function to take a structure (x) and get per-atom gradients used to guide diffusion update

        Inputs:

            xyz (torch.tensor, required): [L,27,3] Coordinates at which the gradient will be computed

        Outputs:

            Ca_grads (torch.tensor): [L,3] The gradient at each Ca atom
        """
        if self.potential_manager is None or self.potential_manager.is_empty():
            return torch.zeros(xyz.shape[0], 3, device=xyz.device, dtype=xyz.dtype)

        xyz.requires_grad_(True)
        if xyz.grad is not None:
            xyz.grad.zero_()

        current_potential = self.potential_manager.compute_all_potentials(xyz)
        current_potential.backward()

        Ca_grads = xyz.grad[:, 1, :].clone()
        xyz.requires_grad_(False) # 计算后分离计算图

        if diffusion_mask is not None:
            Ca_grads[diffusion_mask] = 0

        if torch.isnan(Ca_grads).any():
            self._log.warning("WARNING: NaN in potential gradients, replacing with zero grad.")
            Ca_grads = torch.nan_to_num(Ca_grads)

        return Ca_grads


    
    def batch_get_next_pose(
        self,
        xt,
        px0,
        t,
        diffusion_mask,
        lengths,
        fix_motif=True,
        align_motif=True,
        include_motif_sidechains=True,
    ):
        """
        执行单步反向去噪，从当前噪声状态 xt 和模型对最终结果的预测 px0，计算出上一个时间步的噪声状态 x_t-1。

        主要步骤:
        1. (可选) 将预测出的无噪结构 px0 中的 motif 区域对齐到当前噪声结构 xt 的 motif 区域上，以防止生成过程中的漂移。
        2. 调用 get_next_ca 计算去噪后的 C-alpha 原子坐标，完成平移部分的去噪。
        3. 调用 get_next_frames 计算去噪后的骨架朝向（frames），完成旋转部分的去噪。
        4. (可选) 计算并施加来自 guiding potentials (如接触势、半径势等) 的梯度，对 C-alpha 坐标的更新进行微调。
        5. 组合更新后的平移和旋转，生成最终的 x_t-1 结构。

        Args:
            xt (torch.Tensor): 当前时间步 t 的噪声原子坐标。形状: (N_total, 14, 3) 或 (N_total, 27, 3)，其中 N_total 是蛋白质总长度。
            px0 (torch.Tensor): 去噪模型根据 xt 预测出的在 t=0 时的“干净”结构。形状: (N_total, 14, 3) 或 (N_total, 27, 3)。
            t (int): 当前所处的时间步（从 T 到 1）。
            diffusion_mask (torch.Tensor): 一个布尔掩码，标记哪些残基是固定的 (motif)，不参与扩散。形状: (N_total,)。
                * True 代表该残基是固定的
                * False 代表该残基是需要模型生成的。
            fix_motif (bool, optional): 是否在去噪过程中严格保持 motif 区域的结构不变。如果为 False，则 motif 区域也会参与扩散。默认为 True。
            align_motif (bool, optional): 是否在计算 x_t-1 前，将 px0 的 motif 对齐到 xt 的 motif 上。这是保证 motif 稳定性的关键步骤。默认为 True。
            include_motif_sidechains (bool, optional): 是否在生成的 x_t-1 中保留 motif 区域的侧链原子。默认为 True。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
            - torch.Tensor: 计算得到的上一个时间步的原子坐标 x_t-1。形状: (N_total, 14, 3)。
            - torch.Tensor: 经过对齐（如果 align_motif=True）后的 px0。它将被用于下一个时间步的自条件（self-conditioning）输入。形状: (N_total, 14, 3) 或 (N_total, 27, 3)。
        """
        N_total, n_atom = xt.shape[:2]
        assert n_atom in [14, 27] and px0.shape[1] in [14, 27]

        # 1. (可选) 对齐 px0 到 xt 的 motif
        if align_motif and torch.any(diffusion_mask):
            px0 = self.batch_align_to_xt_motif(px0, xt, diffusion_mask, lengths)

        if not fix_motif:
            diffusion_mask = torch.zeros_like(diffusion_mask)

        # 2. 计算 C-alpha 坐标的更新
        noise_scale_ca = self.noise_schedule_ca(t)
        _, ca_deltas = get_next_ca(
            xt,
            px0,
            t,
            diffusion_mask,
            crd_scale=self.crd_scale,
            beta_schedule=self.schedule,
            alphabar_schedule=self.alphabar_schedule,
            noise_scale=noise_scale_ca,
        )

        # 3. 计算骨架朝向 (frames) 的更新
        noise_scale_frame = self.noise_schedule_frame(t)
        frames_next = batch_get_next_frames(
            xt,
            px0,
            t,
            T=self.T,
            diffuser=self.diffuser,
            so3_type=self.so3_type,
            diffusion_mask=diffusion_mask,
            noise_scale=noise_scale_frame,
        )

        # 4. (可选) 应用指导势能的梯度
        if self.potential_manager is not None and not self.potential_manager.is_empty():
            grad_ca = self.get_potential_gradients(xt.clone(), diffusion_mask=diffusion_mask)
            guide_scale = self.potential_manager.get_guide_scale(t)
            ca_deltas += guide_scale * grad_ca

        # 5. 组合平移和旋转更新
        frames_next = frames_next + ca_deltas.unsqueeze(1)

        # 6. 构建下一时间步的全原子结构
        fullatom_next = torch.full_like(xt, float("nan"))
        fullatom_next[..., :3, :] = frames_next

        if include_motif_sidechains and torch.any(diffusion_mask):
            fullatom_next[diffusion_mask, :14, :] = xt[diffusion_mask, :14, :]

        return fullatom_next[:, :14, :], px0
    


def sampler_selector(conf: DictConfig):
    if conf.scaffoldguided.scaffoldguided:
        sampler = model_runners.ScaffoldedSampler(conf)
    else:
        if conf.inference.model_runner == "default":
            sampler = model_runners.Sampler(conf)
        elif conf.inference.model_runner == "SelfConditioning":
            sampler = model_runners.SelfConditioning(conf)
        elif conf.inference.model_runner == "ScaffoldedSampler":
            sampler = model_runners.ScaffoldedSampler(conf)
        else:
            raise ValueError(f"Unrecognized sampler {conf.model_runner}")
    return sampler


def parse_pdb(filename, **kwargs):
    """extract xyz coords for all heavy atoms"""
    with open(filename,"r") as f:
        lines=f.readlines()
    return parse_pdb_lines(lines, **kwargs)


def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    """
    Returns: out 是一个字典：
            "xyz":
                内容: 蛋白质所有重原子的三维笛卡尔坐标。
                结构: NumPy 数组 (np.ndarray)。
                形状: (L, 14, 3)，其中 L 是去重后的残基数量，14 是标准化的原子数量上限，3 代表 x, y, z 坐标。不存在的原子位置用 0.0 填充。
            "mask":
                内容: 一个布尔掩码，标记 xyz 数组中哪些原子是真实存在于PDB文件中的。
                结构: NumPy 布尔数组 (np.ndarray)。
                形状: (L, 14)。mask[i, j] 为 True 表示第 i 个残基的第 j 个原子坐标是真实的。
            "idx":
                内容: PDB文件中每个残基的原始编号。
                结构: NumPy 整数数组 (np.ndarray)。
                形状: (L,)。例如 [10, 11, 12, ...]。
            "seq":
                内容: 蛋白质的氨基酸序列，已经从三字母码转换成了数字表示。
                结构: NumPy 整数数组 (np.ndarray)。
                形状: (L,)。
            "pdb_idx":
                内容: 每个残基的唯一标识符，由链ID和残基编号组成。
                结构: Python 列表 (list)，每个元素是一个元组 (chain_id, residue_number)。
                形状: 长度为 L 的列表，例如 [('A', 10), ('A', 11), ...]。
            "xyz_het" (可选):
                内容: 如果 parse_hetatom=True，此键存在，包含所有杂原子的三维坐标。
                结构: NumPy 数组 (np.ndarray)。
                形状: (num_het_atoms, 3)。
            "info_het" (可选):
                内容: 如果 parse_hetatom=True，此键存在，包含每个杂原子的详细信息。
                结构: Python 列表 (list)，每个元素是一个字典，如 {'idx': ..., 'atom_id': ..., ...}。
                形状: 长度为 num_het_atoms 的列表。
    """
    # indices of residues observed in the structure
    res, pdb_idx = [],[]
    for l in lines:
        if l[:4] == "ATOM" and l[12:16].strip() == "CA":
            res.append((l[22:26], l[17:20]))
            # chain letter, res num
            pdb_idx.append((l[21:22].strip(), int(l[22:26].strip())))
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    pdb_idx = [
        (l[21:22].strip(), int(l[22:26].strip()))
        for l in lines
        if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = (
            l[21:22],
            int(l[22:26]),
            " " + l[12:16].strip().ljust(3),
            l[17:20],
        )
        if (chain,resNo) in pdb_idx:
            idx = pdb_idx.index((chain, resNo))
            # for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            for i_atm, tgtatm in enumerate(
                util.aa2long[util.aa2num[aa]][:14]
                ):
                if (
                    tgtatm is not None and tgtatm.strip() == atom.strip()
                    ):  # ignore whitespace
                    xyz[idx, i_atm, :] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i, idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]

    seq = np.array(seq)[i_unique]

    out = {
        "xyz": xyz,  # cartesian coordinates, [Lx14]
        "mask": mask,  # mask showing which atoms are present in the PDB file, [Lx14]
        "idx": np.array(
            [i[1] for i in pdb_idx]
        ),  # residue numbers in the PDB file, [L]
        "seq": np.array(seq),  # amino acid sequence, [L]
        "pdb_idx": pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
    }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6] == "HETATM" and not (ignore_het_h and l[77] == "H"):
                info_het.append(
                    dict(
                        idx=int(l[7:11]),
                        atom_id=l[12:16],
                        atom_type=l[77],
                        name=l[16:20],
                    )
                )
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out["xyz_het"] = np.array(xyz_het)
        out["info_het"] = info_het

    return out


def process_target(pdb_path, parse_hetatom=False, center=True):
    """
    读取一个PDB文件，并将其处理成模型所需的标准特征格式。
    """
    # Read target pdb and extract features.
    target_struct = parse_pdb(pdb_path, parse_hetatom=parse_hetatom)

    # Zero-center positions
    ca_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True)
    if not center:
        ca_center = 0
    xyz = torch.from_numpy(target_struct["xyz"] - ca_center)
    seq_orig = torch.from_numpy(target_struct["seq"])
    atom_mask = torch.from_numpy(target_struct["mask"])
    seq_len = len(xyz)

    # Make 27 atom representation
    xyz_27 = torch.full((seq_len, 27, 3), np.nan).float()
    xyz_27[:, :14, :] = xyz[:, :14, :]

    mask_27 = torch.full((seq_len, 27), False)
    mask_27[:, :14] = atom_mask
    out = {
        "xyz_27": xyz_27,
        "mask_27": mask_27,
        "seq": seq_orig,
        "pdb_idx": target_struct["pdb_idx"],
    }
    if parse_hetatom:
        out["xyz_het"] = target_struct["xyz_het"]
        out["info_het"] = target_struct["info_het"]
    return out


def get_idx0_hotspots(mappings, ppi_conf, binderlen):
    """
    Take pdb-indexed hotspot resudes and the length of the binder, and makes the 0-indexed tensor of hotspots
    """

    hotspot_idx = None
    if binderlen > 0:
        if ppi_conf.hotspot_res is not None:
            assert all(
                [i[0].isalpha() for i in ppi_conf.hotspot_res]
            ), "Hotspot residues need to be provided in pdb-indexed form. E.g. A100,A103"
            hotspots = [(i[0], int(i[1:])) for i in ppi_conf.hotspot_res]
            hotspot_idx = []
            for i, res in enumerate(mappings["receptor_con_ref_pdb_idx"]):
                if res in hotspots:
                    hotspot_idx.append(mappings["receptor_con_hal_idx0"][i])
    return hotspot_idx


class BlockAdjacency:
    """
    Class for handling PPI design inference with ss/block_adj inputs.
    Basic idea is to provide a list of scaffolds, and to output ss and adjacency
    matrices based off of these, while sampling additional lengths.
    Inputs:
        - scaffold_list: list of scaffolds (e.g. ['2kl8','1cif']). Can also be a .txt file.
        - scaffold dir: directory where scaffold ss and adj are precalculated
        - sampled_insertion: how many additional residues do you want to add to each loop segment? Randomly sampled 0-this number (or within given range)
        - sampled_N: randomly sample up to this number of additional residues at N-term
        - sampled_C: randomly sample up to this number of additional residues at C-term
        - ss_mask: how many residues do you want to mask at either end of a ss (H or E) block. Fixed value
        - num_designs: how many designs are you wanting to generate? Currently only used for bookkeeping
        - systematic: do you want to systematically work through the list of scaffolds, or randomly sample (default)
        - num_designs_per_input: Not really implemented yet. Maybe not necessary
    Outputs:
        - L: new length of chain to be diffused
        - ss: all loops and insertions, and ends of ss blocks (up to ss_mask) set to mask token (3). Onehot encoded. (L,4)
        - adj: block adjacency with equivalent masking as ss (L,L)
    """

    def __init__(self, conf, num_designs):
        """
        Parameters:
          inputs:
             conf.scaffold_list as conf
             conf.inference.num_designs for sanity checking
        """
       
        self.conf=conf 
        # either list or path to .txt file with list of scaffolds
        if self.conf.scaffoldguided.scaffold_list is not None:
            if type(self.conf.scaffoldguided.scaffold_list) == list:
                self.scaffold_list = scaffold_list
            elif self.conf.scaffoldguided.scaffold_list[-4:] == ".txt":
                # txt file with list of ids
                list_from_file = []
                with open(self.conf.scaffoldguided.scaffold_list, "r") as f:
                    for line in f:
                        list_from_file.append(line.strip())
                self.scaffold_list = list_from_file
            else:
                raise NotImplementedError
        else:
            self.scaffold_list = [
                os.path.split(i)[1][:-6]
                for i in glob.glob(f"{self.conf.scaffoldguided.scaffold_dir}/*_ss.pt")
            ]
            self.scaffold_list.sort()

        # path to directory with scaffolds, ss files and block_adjacency files
        self.scaffold_dir = self.conf.scaffoldguided.scaffold_dir

        # maximum sampled insertion in each loop segment
        if "-" in str(self.conf.scaffoldguided.sampled_insertion):
            self.sampled_insertion = [
                int(str(self.conf.scaffoldguided.sampled_insertion).split("-")[0]),
                int(str(self.conf.scaffoldguided.sampled_insertion).split("-")[1]),
            ]
        else:
            self.sampled_insertion = [0, int(self.conf.scaffoldguided.sampled_insertion)]

        # maximum sampled insertion at N- and C-terminus
        if "-" in str(self.conf.scaffoldguided.sampled_N):
            self.sampled_N = [
                int(str(self.conf.scaffoldguided.sampled_N).split("-")[0]),
                int(str(self.conf.scaffoldguided.sampled_N).split("-")[1]),
            ]
        else:
            self.sampled_N = [0, int(self.conf.scaffoldguided.sampled_N)]
        if "-" in str(self.conf.scaffoldguided.sampled_C):
            self.sampled_C = [
                int(str(self.conf.scaffoldguided.sampled_C).split("-")[0]),
                int(str(self.conf.scaffoldguided.sampled_C).split("-")[1]),
            ]
        else:
            self.sampled_C = [0, int(self.conf.scaffoldguided.sampled_C)]

        # number of residues to mask ss identity of in H/E regions (from junction)
        # e.g. if ss_mask = 2, L,L,L,H,H,H,H,H,H,H,L,L,E,E,E,E,E,E,L,L,L,L,L,L would become\
        # M,M,M,M,M,H,H,H,M,M,M,M,M,M,E,E,M,M,M,M,M,M,M,M where M is mask
        self.ss_mask = self.conf.scaffoldguided.ss_mask

        # whether or not to work systematically through the list
        self.systematic = self.conf.scaffoldguided.systematic

        self.num_designs = num_designs

        if len(self.scaffold_list) > self.num_designs:
            print(
                "WARNING: Scaffold set is bigger than num_designs, so not every scaffold type will be sampled"
            )

        # for tracking number of designs
        self.num_completed = 0
        if self.systematic:
            self.item_n = 0

        # whether to mask loops or not
        if not self.conf.scaffoldguided.mask_loops:
            assert self.conf.scaffoldguided.sampled_N == 0, "can't add length if not masking loops"
            assert self.conf.scaffoldguided.sampled_C == 0, "can't add lemgth if not masking loops"
            assert self.conf.scaffoldguided.sampled_insertion == 0, "can't add length if not masking loops"
            self.mask_loops = False
        else:
            self.mask_loops = True

    def get_ss_adj(self, item):
        """
        Given at item, get the ss tensor and block adjacency matrix for that item
        """
        ss = torch.load(os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_ss.pt'))
        adj = torch.load(
            os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_adj.pt')
        )

        return ss, adj

    def mask_to_segments(self, mask):
        """
        Takes a mask of True (loop) and False (non-loop), and outputs list of tuples (loop or not, length of element)
        """
        segments = []
        begin = -1
        end = -1
        for i in range(mask.shape[0]):
            # Starting edge case
            if i == 0:
                begin = 0
                continue

            if not mask[i] == mask[i - 1]:
                end = i
                if mask[i - 1].item() is True:
                    segments.append(("loop", end - begin))
                else:
                    segments.append(("ss", end - begin))
                begin = i

        # Ending edge case: last segment is length one
        if not end == mask.shape[0]:
            if mask[i].item() is True:
                segments.append(("loop", mask.shape[0] - begin))
            else:
                segments.append(("ss", mask.shape[0] - begin))
        return segments

    def expand_mask(self, mask, segments):
        """
        Function to generate a new mask with dilated loops and N and C terminal additions
        """
        N_add = random.randint(self.sampled_N[0], self.sampled_N[1])
        C_add = random.randint(self.sampled_C[0], self.sampled_C[1])

        output = N_add * [False]
        for ss, length in segments:
            if ss == "ss":
                output.extend(length * [True])
            else:
                # randomly sample insertion length
                ins = random.randint(
                    self.sampled_insertion[0], self.sampled_insertion[1]
                )
                output.extend((length + ins) * [False])
        output.extend(C_add * [False])
        assert torch.sum(torch.tensor(output)) == torch.sum(~mask)
        return torch.tensor(output)

    def expand_ss(self, ss, adj, mask, expanded_mask):
        """
        Given an expanded mask, populate a new ss and adj based on this
        """
        ss_out = torch.ones(expanded_mask.shape[0]) * 3  # set to mask token
        adj_out = torch.full((expanded_mask.shape[0], expanded_mask.shape[0]), 0.0)
        ss_out[expanded_mask] = ss[~mask]
        expanded_mask_2d = torch.full(adj_out.shape, True)
        # mask out loops/insertions, which is ~expanded_mask
        expanded_mask_2d[~expanded_mask, :] = False
        expanded_mask_2d[:, ~expanded_mask] = False

        mask_2d = torch.full(adj.shape, True)
        # mask out loops. This mask is True=loop
        mask_2d[mask, :] = False
        mask_2d[:, mask] = False
        adj_out[expanded_mask_2d] = adj[mask_2d]
        adj_out = adj_out.reshape((expanded_mask.shape[0], expanded_mask.shape[0]))

        return ss_out, adj_out

    def mask_ss_adj(self, ss, adj, expanded_mask):
        """
        Given an expanded ss and adj, mask some number of residues at either end of non-loop ss
        """
        original_mask = torch.clone(expanded_mask)
        if self.ss_mask > 0:
            for i in range(1, self.ss_mask + 1):
                expanded_mask[i:] *= original_mask[:-i]
                expanded_mask[:-i] *= original_mask[i:]

        if self.mask_loops:
            ss[~expanded_mask] = 3
            adj[~expanded_mask, :] = 0
            adj[:, ~expanded_mask] = 0

        # mask adjacency
        adj[~expanded_mask] = 2
        adj[:, ~expanded_mask] = 2

        return ss, adj

    def get_scaffold(self):
        """
        Wrapper method for pulling an item from the list, and preparing ss and block adj features
        """
        
        # Handle determinism. Useful for integration tests
        if self.conf.inference.deterministic:
            torch.manual_seed(self.num_completed)
            np.random.seed(self.num_completed)
            random.seed(self.num_completed)
  
        if self.systematic:
            # reset if num designs > num_scaffolds
            if self.item_n >= len(self.scaffold_list):
                self.item_n = 0
            item = self.scaffold_list[self.item_n]
            self.item_n += 1
        else:
            item = random.choice(self.scaffold_list)
        print("Scaffold constrained based on file: ", item)
        # load files
        ss, adj = self.get_ss_adj(item)
        adj_orig = torch.clone(adj)
        # separate into segments (loop or not)
        mask = torch.where(ss == 2, 1, 0).bool()
        segments = self.mask_to_segments(mask)

        # insert into loops to generate new mask
        expanded_mask = self.expand_mask(mask, segments)

        # expand ss and adj
        ss, adj = self.expand_ss(ss, adj, mask, expanded_mask)

        # finally, mask some proportion of the ss at either end of the non-loop ss blocks
        ss, adj = self.mask_ss_adj(ss, adj, expanded_mask)

        # and then update num_completed
        self.num_completed += 1

        return ss.shape[0], torch.nn.functional.one_hot(ss.long(), num_classes=4), adj


class Target:
    """
    Class to handle targets (fixed chains).
    Inputs:
        - path to pdb file
        - hotspot residues, in the form B10,B12,B60 etc
        - whether or not to crop, and with which method
    Outputs:
        - Dictionary of xyz coordinates, indices, pdb_indices, pdb mask
    """

    def __init__(self, conf: DictConfig, hotspots=None):
        self.pdb = parse_pdb(conf.target_path)

        if hotspots is not None:
            self.hotspots = hotspots
        else:
            self.hotspots = []
        self.pdb["hotspots"] = np.array(
            [
                True if f"{i[0]}{i[1]}" in self.hotspots else False
                for i in self.pdb["pdb_idx"]
            ]
        )

        if conf.contig_crop:
            self.contig_crop(conf.contig_crop)

    def parse_contig(self, contig_crop):
        """
        Takes contig input and parses
        """
        contig_list = []
        for contig in contig_crop[0].split(" "):
            subcon = []
            for crop in contig.split("/"):
                if crop[0].isalpha():
                    subcon.extend(
                        [
                            (crop[0], p)
                            for p in np.arange(
                                int(crop.split("-")[0][1:]), int(crop.split("-")[1]) + 1
                            )
                        ]
                    )
            contig_list.append(subcon)
        return contig_list

    def contig_crop(self, contig_crop, residue_offset=200) -> None:
        """
        Method to take a contig string referring to the receptor and output a pdb dictionary with just this crop
        NB there are two ways to provide inputs:
            - 1) e.g. B1-30,0 B50-60,0. This will add a residue offset between each chunk
            - 2) e.g. B1-30,B50-60,B80-100. This will keep the original indexing of the pdb file.
        Can handle the target being on multiple chains
        """

        # add residue offset between chains if multiple chains in receptor file
        for idx, val in enumerate(self.pdb["pdb_idx"]):
            if idx != 0 and val != self.pdb["pdb_idx"][idx - 1]:
                self.pdb["idx"][idx:] += residue_offset + idx

        # convert contig to mask
        contig_list = self.parse_contig(contig_crop)

        # add residue offset to different parts of contig_list
        for contig in contig_list[1:]:
            start = int(contig[0][1])
            self.pdb["idx"][start:] += residue_offset
        # flatten list
        contig_list = [i for j in contig_list for i in j]
        mask = np.array(
            [True if i in contig_list else False for i in self.pdb["pdb_idx"]]
        )

        # sanity check
        assert np.sum(self.pdb["hotspots"]) == np.sum(
            self.pdb["hotspots"][mask]
        ), "Supplied hotspot residues are missing from the target contig!"
        # crop pdb
        for key, val in self.pdb.items():
            try:
                self.pdb[key] = val[mask]
            except:
                self.pdb[key] = [i for idx, i in enumerate(val) if mask[idx]]
        self.pdb["crop_mask"] = mask

    def get_target(self):
        return self.pdb

def ss_from_contig(ss_masks: dict):
    """  
    Function for taking 1D masks for each of the ss types, and outputting a secondary structure input
    """
    L=len(ss_masks['helix'])
    ss=torch.zeros((L, 4)).long()
    ss[:,3] = 1 #mask
    for idx, mask in enumerate([ss_masks['helix'],ss_masks['strand'], ss_masks['loop']]):
        ss[mask,idx] = 1
        ss[mask, 3] = 0 # remove the mask token
    return ss