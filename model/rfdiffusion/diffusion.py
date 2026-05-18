# script for diffusion protocols
import torch
import pickle
import numpy as np
import os
import logging

from typing import Optional, Tuple

from scipy.spatial.transform import Rotation as scipy_R

from model.rfdiffusion.util import rigid_from_3_points

from model.rfdiffusion.util_module import ComputeAllAtomCoords

from model.rfdiffusion import igso3
import time

torch.set_printoptions(sci_mode=False)


def get_beta_schedule(T, b0, bT, schedule_type, schedule_params={}, inference=False):
    """
    Given a noise schedule type, create the beta schedule
    """
    assert schedule_type in ["linear"]

    # Adjust b0 and bT if T is not 200
    # This is a good approximation, with the beta correction below, unless T is very small
    # assert T >= 15, "With discrete time and T < 15, the schedule is badly approximated"
    b0 *= 200 / T
    bT *= 200 / T

    # linear noise schedule
    if schedule_type == "linear":
        schedule = torch.linspace(b0, bT, T)

    else:
        raise NotImplementedError(f"Schedule of type {schedule_type} not implemented.")

    # get alphabar_t for convenience
    alpha_schedule = 1 - schedule
    alphabar_t_schedule = torch.cumprod(alpha_schedule, dim=0)

    # if inference:
    #     print(
    #         f"With this beta schedule ({schedule_type} schedule, beta_0 = {round(b0, 3)}, beta_T = {round(bT,3)}), alpha_bar_T = {alphabar_t_schedule[-1]}"
    #     )

    return schedule, alpha_schedule, alphabar_t_schedule


class EuclideanDiffuser:
    # class for diffusing points in 3D

    def __init__(
        self,
        T,
        b_0,
        b_T,
        schedule_type="linear",
        schedule_kwargs={},
        device='cpu'
    ):
        self.T = T

        # make noise/beta schedule
        (
            self.beta_schedule,
            self.alpha_schedule,
            self.alphabar_schedule,
        ) = get_beta_schedule(T, b_0, b_T, schedule_type, **schedule_kwargs)

        self.beta_schedule = self.beta_schedule.to(device)
        self.alpha_schedule = self.alpha_schedule.to(device)
        self.alphabar_schedule = self.alphabar_schedule.to(device)

    def apply_kernel(self, x, t, diffusion_mask=None, var_scale=1):
        """
        Applies a noising kernel to the points in x

        Args:
            x (torch.tensor, required): (N_total,3,3) set of backbone coordinates
            t (int, required): Which timestep, 1-base
            diffusion_mask (torch.tensor): (N_total,) set of 0/1. True/1 is NOT diffused, False/0 IS diffused
            noise_scale (float): scale for noise
        Returns:
            out_crds: (N_total,3,3), set of backbone coordinates after one step of diffusion
            delta: (N_total,3,3), the change in coordinates between inpuy x and out_crds
        """
        t_idx = t - 1  # 从1-based索引转为0-based索引
        if t_idx < 0 or t_idx >= len(self.alphabar_schedule):
            raise IndexError(f"时间步 t={t} 超出有效范围 [1, {self.T}]")
        assert len(x.shape) == 3

        # C-alpha 坐标ca_xyz: (N_total, 3)
        ca_xyz = x[:, 1, :]

        # 从 schedule 中获取 alphabar_t
        alphabar_t = self.alphabar_schedule[t_idx].to(device=x.device, dtype=x.dtype)

        # 计算 q(x_t | x_0) 的均值和方差
        # q(x_t | x_0) = N(x_t; sqrt(alphabar_t) * x_0, (1 - alphabar_t) * I)
        mean = torch.sqrt(alphabar_t) * ca_xyz
        var = (1 - alphabar_t) * var_scale
        
        # 从高斯分布中采样得到加噪后的坐标，torch.normal 的第二个参数是标准差
        sampled_crds = torch.normal(mean, torch.sqrt(var))
        
        # 计算坐标变化量
        delta = sampled_crds - ca_xyz

        # 应用 diffusion mask，不扩散的区域 delta 为 0
        if diffusion_mask is not None:
            if not diffusion_mask.dtype == torch.bool:
                diffusion_mask = diffusion_mask.bool()
            delta[diffusion_mask, ...] = 0

        # 将变化量应用到整个骨架
        out_crds = x + delta[:, None, :]

        return out_crds, delta
    
    

def write_pkl(save_path: str, pkl_data):
    """Serialize data into a pickle file."""
    with open(save_path, "wb") as handle:
        pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=False):
    """Read data from a pickle file."""
    with open(read_path, "rb") as handle:
        try:
            return pickle.load(handle)
        except Exception as e:
            if verbose:
                print(f"Failed to read {read_path}")
            raise (e)


class IGSO3:
    """
    Class for taking in a set of backbone crds and performing IGSO3 diffusion
    on all of them.

    Unlike the diffusion on translations, much of this class is written for a
    scaling between an initial time t=0 and final time t=1.
    """

    def __init__(
        self,
        *,
        T,
        min_sigma,
        max_sigma,
        min_b,
        max_b,
        cache_dir,
        num_omega=1000,
        schedule="linear",
        L=2000,
        device='cpu'
    ):
        """

        Args:
            T: total number of time steps
            min_sigma: smallest allowed scale parameter, should be at least 0.01 to maintain numerical stability.  Recommended value is 0.05.
            max_sigma: for exponential schedule, the largest scale parameter. Ignored for recommeded linear schedule
            min_b: lower value of beta in Ho schedule analogue
            max_b: upper value of beta in Ho schedule analogue
            num_omega: discretization level in the angles across [0, pi]
            schedule: currently only linear and exponential are supported.  The exponential schedule may be noising too slowly.
            L: truncation level
        """
        self._log = logging.getLogger(__name__)

        self.T = T
        self.device = device

        self.schedule = schedule
        self.cache_dir = cache_dir
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        if self.schedule == "linear":
            self.min_b = min_b
            self.max_b = max_b
            self.max_sigma = self.sigma(1.0)
        self.num_omega = num_omega
        self.num_sigma = 500
        # Calculate igso3 values.
        self.L = L  # truncation level
        self.igso3_vals = self._calc_igso3_vals(L=L)
        self.step_size = 1 / self.T

        self.igso3_vals_torch = {
            "cdf": torch.from_numpy(self.igso3_vals['cdf']).to(self.device),
            "discrete_omega": torch.from_numpy(self.igso3_vals['discrete_omega']).to(self.device)
        }

    def _calc_igso3_vals(self, L=2000):
        """_calc_igso3_vals computes numerical approximations to the
        relevant analytically intractable functionals of the igso3
        distribution.

        The calculated values are cached, or loaded from cache if they already
        exist.

        Args:
            L: truncation level for power series expansion of the pdf.
        """
        replace_period = lambda x: str(x).replace(".", "_")
        if self.schedule == "linear":
            cache_fname = os.path.join(
                self.cache_dir,
                f"T_{self.T}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}"
                + f"_min_b_{replace_period(self.min_b)}_max_b_{replace_period(self.max_b)}_schedule_{self.schedule}.pkl",
            )
        elif self.schedule == "exponential":
            cache_fname = os.path.join(
                self.cache_dir,
                f"T_{self.T}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}"
                f"_max_sigma_{replace_period(self.max_sigma)}_schedule_{self.schedule}",
            )
        else:
            raise ValueError(f"Unrecognize schedule {self.schedule}")

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        if os.path.exists(cache_fname):
            self._log.info("Using cached IGSO3.")
            igso3_vals = read_pkl(cache_fname)
        else:
            self._log.info("Calculating IGSO3.")
            igso3_vals = igso3.calculate_igso3(
                num_sigma=self.num_sigma,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_omega=self.num_omega
            )
            write_pkl(cache_fname, igso3_vals)

        return igso3_vals

    @property
    def discrete_sigma(self):
        return self.igso3_vals["discrete_sigma"]

    def sigma_idx(self, sigma: np.ndarray):
        """
        Calculates the index for discretized sigma during IGSO(3) initialization."""
        return np.digitize(sigma, self.discrete_sigma) - 1

    def t_to_idx(self, t: np.ndarray):
        """
        Helper function to go from discrete time index t to corresponding sigma_idx.

        Args:
            t: time index (integer between 1 and 200)
        """
        continuous_t = t / self.T
        return self.sigma_idx(self.sigma(continuous_t))

    def sigma(self, t: torch.tensor):
        """
        Extract \sigma(t) corresponding to chosen sigma schedule.

        Args:
            t: torch tensor with time between 0 and 1
        """
        if not type(t) == torch.Tensor:
            t = torch.tensor(t)
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        if self.schedule == "exponential":
            sigma = t * np.log10(self.max_sigma) + (1 - t) * np.log10(self.min_sigma)
            return 10**sigma
        elif self.schedule == "linear":  # Variance exploding analogue of Ho schedule
            # add self.min_sigma for stability
            return (
                self.min_sigma
                + t * self.min_b
                + (1 / 2) * (t**2) * (self.max_b - self.min_b)
            )
        else:
            raise ValueError(f"Unrecognize schedule {self.schedule}")

    def g(self, t):
        """
        g returns the drift coefficient at time t

        since
            sigma(t)^2 := \int_0^t g(s)^2 ds,
        for arbitrary sigma(t) we invert this relationship to compute
            g(t) = sqrt(d/dt sigma(t)^2).

        Args:
            t: scalar time between 0 and 1

        Returns:
            drift cooeficient as a scalar.
        """
        t = torch.tensor(t, requires_grad=True)
        sigma_sqr = self.sigma(t) ** 2
        grads = torch.autograd.grad(sigma_sqr.sum(), t)[0]
        return torch.sqrt(grads)



    def sample(self, ts, n_samples=1):
        """
        sample uses the inverse cdf to sample an angle of rotation from
        IGSO(3)

        Args:
            ts: array of integer time steps to sample from.
            n_samples: number of samples to draw.
        Returns:
            sampled angles of rotation. [len(ts), N]
        """
        assert all(t > 0 for t in ts), "时间步必须是1-indexed，不能为零"
        
        # 假定所有样本在同一步（或ts列表只有一个元素），这是我们当前用例
        t = ts[0] 
        sigma_idx = self.t_to_idx(t)

        cdf = self.igso3_vals_torch["cdf"][sigma_idx]       # 在 GPU 上获取预计算的 CDF值
        omegas = self.igso3_vals_torch["discrete_omega"]    # 在 GPU 上获取预计算的 omega 值
        u = torch.rand(n_samples, device=self.device)       # 在 GPU 上生成均匀分布的随机数

        # 使用 torch.searchsorted 找到每个随机数 u 在 cdf 中的插入位置
        # 'right=True' 使得结果等价于 np.searchsorted(side='right')
        # 这会告诉我们 u 位于 cdf 的哪个区间 (indices-1, indices)
        indices = torch.searchsorted(cdf, u, right=True)

        # 处理边界情况，防止索引越界
        idx_left = (indices - 1).clamp(min=0)
        idx_right = indices.clamp(max=len(cdf) - 1)

        # 获取区间左右两端的 cdf 值和 omega 值
        cdf_left = cdf[idx_left]
        cdf_right = cdf[idx_right]
        omega_left = omegas[idx_left]
        omega_right = omegas[idx_right]
        
        # 计算插值权重
        # 添加一个极小值避免分母为零
        denominator = cdf_right - cdf_left
        weight = (u - cdf_left) / denominator.clamp(min=1e-9)
        
        # 处理 cdf 值相等的情况，此时权重应为0
        weight[denominator < 1e-9] = 0

        # 进行线性插值计算最终的采样角度
        sampled_omegas = omega_left + weight * (omega_right - omega_left)

        # 返回与原始函数形状兼容的张量
        return sampled_omegas.unsqueeze(0)


    def sample_vec(self, ts, n_samples=1):
        """
        sample_vec generates a rotation vector(s) from IGSO(3) at time steps ts.

        Return:
            Sampled vector of shape [len(ts), N, 3]
        """
        # x: shape [len(ts), n_samples, 3]
        x = torch.randn(len(ts), n_samples, 3, device=self.device)
        
        # x: shape [len(ts), n_samples, 3]（归一化后形状不变）
        x = torch.nn.functional.normalize(x, dim=-1)

        # self.sample(...) 返回旋转角度 [len(ts), n_samples]
        # 将旋转轴和旋转角相乘得到旋转向量
        return x * self.sample(ts, n_samples=n_samples).unsqueeze(-1)


    def score_norm(self, t, omega):
        """
        score_norm computes the score norm based on the time step and angle
        Args:
            t: integer time step
            omega: angles (scalar or shape [N])
        Return:
            score_norm with same shape as omega
        """
        sigma_idx = self.t_to_idx(t)
        score_norm_t = np.interp(
            omega,
            self.igso3_vals["discrete_omega"],
            self.igso3_vals["score_norm"][sigma_idx],
        )
        return score_norm_t

    def score_vec(self, ts, vec):
        """score_vec computes the score of the IGSO(3) density as a rotation
        vector. This score vector is in the direction of the sampled vector,
        and has magnitude given by score_norms.

        In particular, Rt @ hat(score_vec(ts, vec)) is what is referred to as
        the score approximation in Algorithm 1


        Args:
            ts: times of shape [T]
            vec: where to compute the score of shape [T, N, 3]
        Returns:
            score vectors of shape [T, N, 3]
        """
        omega = np.linalg.norm(vec, axis=-1)
        all_score_norm = []
        for i, t in enumerate(ts):
            omega_t = omega[i]
            t_idx = t - 1
            sigma_idx = self.t_to_idx(t)
            score_norm_t = np.interp(
                omega_t,
                self.igso3_vals["discrete_omega"],
                self.igso3_vals["score_norm"][sigma_idx],
            )[:, None]
            all_score_norm.append(score_norm_t)
        score_norm = np.stack(all_score_norm, axis=0)
        return score_norm * vec / omega[..., None]

    def exp_score_norm(self, ts):
        """exp_score_norm returns the expected value of norm of the score for
        IGSO(3) with time parameter ts of shape [T].
        """
        sigma_idcs = [self.t_to_idx(t) for t in ts]
        return self.igso3_vals["exp_score_norms"][sigma_idcs]



    def diffuse_frames(self, xyz: torch.Tensor, final_t: int, diffusion_mask: torch.Tensor = None):
        """diffuse_frames samples from the IGSO(3) distribution to noise frames

        Parameters:
            xyz (torch.tensor, required): (N_total,3,3) set of backbone coordinates
            final_t(int, required): Which timestep
            diffusion_mask (torch.tensor, required): (N_total,) set of bools. True/1 is NOT diffused, False/0 IS diffused
        Returns:
            perturbed_crds：(N_total,3,3), set of backbone coordinates after diffusion
            R_perturbed：(N_total,3,3), set of rotation matrices after diffusion
        """

        device = xyz.device
        dtype = xyz.dtype
        num_res = xyz.shape[0]


        # 1. 从骨架坐标计算初始的刚体变换（旋转矩阵 R_true 和中心 Ca）
        N = xyz[:, 0, :]
        Ca = xyz[:, 1, :]
        C = xyz[:, 2, :]
        R_true, Ca_new = rigid_from_3_points(N[None], Ca[None], C[None])    # R_true: (1, N_total, 3, 3), Ca_new: (1, N_total, 3)
        R_true = R_true.squeeze(0)  # (N_total, 3, 3)
        Ca_new = Ca_new.squeeze(0)  # (N_total, 3)


        # 2. 从IGSO(3)分布中采样旋转向量，为单个时间步 final_t 采样 N_total 个旋转向量
        sampled_rots_np = self.sample_vec([final_t], n_samples=num_res)     # sampled_rots_np: torch.Tensor, [1, N_total, 3]
        sampled_rots = sampled_rots_np.squeeze(0).to(device, dtype=dtype)   # (N_total, 3)


        # 3. 应用diffusion_mask。diffusion_mask 中为 True 的位置不进行旋转，将其旋转向量置零
        if diffusion_mask is not None:
            non_diffusion_mask = ~diffusion_mask.unsqueeze(-1)              # non_diffusion_mask: (N_total, 1), True 的位置需要扩散
            sampled_rots = sampled_rots * non_diffusion_mask

        # 4. 将旋转向量转换为旋转矩阵，使用 Rodrigues' rotation formula 避免 scipy 和 CPU 转换
        R_sampled = self.rodrigues_formula(sampled_rots)                         # R_sampled：(N_total, 3, 3)

        # 5. 计算加噪后的旋转矩阵，R_perturbed = R_sampled @ R_true
        R_perturbed = torch.einsum("nij,njk->nik", R_sampled, R_true)

        # 6. 计算加噪后的坐标，将原始坐标中心化，应用旋转，然后平移回去，perturbed_crds = R_sampled @ (xyz - Ca) + Ca
        xyz_centered = xyz - Ca_new.unsqueeze(1)
        perturbed_crds = torch.einsum("nij,naj->nai", R_sampled, xyz_centered) + Ca_new.unsqueeze(1)

        return perturbed_crds, R_perturbed


    def rodrigues_formula(self, rotvecs: torch.Tensor) -> torch.Tensor:
        """
        将一批旋转向量转换为旋转矩阵。
        (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)

        Args:
            rotvecs (torch.Tensor): 形状为 (N, 3) 的旋转向量。

        Returns:
            torch.Tensor: 形状为 (N, 3, 3) 的旋转矩阵。
        """
        theta = torch.linalg.norm(rotvecs, dim=-1, keepdim=True)
        
        # 为零旋转向量创建一个掩码，以避免除以零
        is_zero = theta.squeeze(-1) < 1e-8
        
        # 归一化以获得旋转轴
        axis = torch.nn.functional.normalize(rotvecs, dim=-1)

        # 构造轴的斜对称矩阵 (cross-product matrix) K
        K = torch.zeros((rotvecs.shape[0], 3, 3), device=rotvecs.device, dtype=rotvecs.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] =  axis[:, 1]
        K[:, 1, 0] =  axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] =  axis[:, 0]

        I = torch.eye(3, device=rotvecs.device, dtype=rotvecs.dtype).expand_as(K)
        
        sin_theta = torch.sin(theta).unsqueeze(-1)
        cos_theta = torch.cos(theta).unsqueeze(-1)

        # Rodrigues' formula
        R = I + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)

        # 对于零旋转，直接使用单位矩阵
        R[is_zero] = torch.eye(3, device=rotvecs.device, dtype=rotvecs.dtype)

        return R

    def reverse_sample_vectorized(
        self, R_t, R_0, t, noise_level, mask=None, return_perturb=False
    ):
        """reverse_sample uses an approximation to the IGSO3 score to sample
        a rotation at the previous time step.

        Roughly - this update follows the reverse time SDE for Reimannian
        manifolds proposed by de Bortoli et al. Theorem 1 [1]. But with an
        approximation to the score based on the prediction of R0.
        Unlike in reference [1], this diffusion on SO(3) relies on geometric
        variance schedule.  Specifically we follow [2] (appendix C) and assume
            sigma_t = sigma_min * (sigma_max / sigma_min)^{t/T},
        for time step t.  When we view this as a discretization  of the SDE
        from time 0 to 1 with step size (1/T).  Following Eq. 5 and Eq. 6,
        this maps on to the forward  time SDEs
            dx = g(t) dBt [FORWARD]
        and
            dx = g(t)^2 score(xt, t)dt + g(t) B't, [REVERSE]
        where g(t) = sigma_t * sqrt(2 * log(sigma_max/ sigma_min)), and Bt and
        B't are Brownian motions. The formula for g(t) obtains from equation 9
        of [2], from which this sampling function may be generalized to
        alternative noising schedules.
        Args:
            R_t: noisy rotation of shape [N, 3, 3]
            R_0: prediction of un-noised rotation
            t: integer time step
            noise_level: scaling on the noise added when obtaining sample
                (preliminary performance seems empirically better with noise
                level=0.5)
            mask: whether the residue is to be updated.  A value of 1 means the
                rotation is not updated from r_t.  A value of 0 means the
                rotation is updated.
        Return:
            sampled rotation matrix for time t-1 of shape [3, 3]
        Reference:
        [1] De Bortoli, V., Mathieu, E., Hutchinson, M., Thornton, J., Teh, Y.
        W., & Doucet, A. (2022). Riemannian score-based generative modeling.
        arXiv preprint arXiv:2202.02763.
        [2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S.,
        & Poole, B. (2020). Score-based generative modeling through stochastic
        differential equations. arXiv preprint arXiv:2011.13456.
        """
        # compute rotation vector corresponding to prediction of how r_t goes to r_0
        R_0, R_t = torch.tensor(R_0), torch.tensor(R_t)
        R_0t = torch.einsum("...ij,...kj->...ik", R_t, R_0)
        R_0t_rotvec = torch.tensor(
            scipy_R.from_matrix(R_0t.cpu().numpy()).as_rotvec()
        ).to(R_0.device)

        # Approximate the score based on the prediction of R0.
        # R_t @ hat(Score_approx) is the score approximation in the Lie algebra
        # SO(3) (i.e. the output of Algorithm 1)
        Omega = torch.linalg.norm(R_0t_rotvec, axis=-1).numpy()
        Score_approx = R_0t_rotvec * (self.score_norm(t, Omega) / Omega)[:, None]

        # Compute scaling for score and sampled noise (following Eq 6 of [2])
        continuous_t = t / self.T
        rot_g = self.g(continuous_t).to(Score_approx.device)

        # Sample and scale noise to add to the rotation perturbation in the
        # SO(3) tangent space.  Since IG-SO(3) is the Brownian motion on SO(3)
        # (up to a deceleration of time by a factor of two), for small enough
        # time-steps, this is equivalent to perturbing r_t with IG-SO(3) noise.
        # See e.g. Algorithm 1 of De Bortoli et al.
        Z = np.random.normal(size=(R_0.shape[0], 3))
        Z = torch.from_numpy(Z).to(Score_approx.device)
        Z *= noise_level

        Delta_r = (rot_g**2) * self.step_size * Score_approx

        # Sample perturbation from discretized SDE (following eq. 6 of [2]),
        # This approximate sampling from IGSO3(* ; Delta_r, rot_g^2 *
        # self.step_size) with tangent Gaussian.
        Perturb_tangent = Delta_r + rot_g * np.sqrt(self.step_size) * Z
        if mask is not None:
            Perturb_tangent *= (1 - mask.long())[:, None, None]
        Perturb = igso3.Exp(Perturb_tangent)

        if return_perturb:
            return Perturb

        Interp_rot = torch.einsum("...ij,...jk->...ik", Perturb, R_t)

        return Interp_rot


class Diffuser:
    # wrapper for yielding diffused coordinates

    def __init__(
        self,
        T,
        b_0,
        b_T,
        min_sigma,
        max_sigma,
        min_b,
        max_b,
        schedule_type,
        so3_schedule_type,
        so3_type,
        crd_scale,
        schedule_kwargs={},
        var_scale=1.0,
        cache_dir=".",
        partial_T=None,
        truncation_level=2000,
        device='cpu'
    ):
        """
        初始化 Diffuser 对象，该对象封装了蛋白质姿态（位置和方向）的`前向扩散过程`。
        它结合了两种独立的扩散器来分别处理平移和旋转：
        1. EuclideanDiffuser: 用于对骨架原子的三维坐标施加高斯噪声，模拟平移扩散。
        2. IGSO3: 用于对骨架的刚体姿态（旋转）施加 SO(3) 流形上的噪声，模拟旋转扩散。

        Args:
            T (int): 扩散/去噪过程的总步数。
            b_0 (float): 欧几里得（平移）噪声调度表的起始 beta 值。
            b_T (float): 欧几里得（平移）噪声调度表的结束 beta 值。
            min_sigma (float): IGSO3（旋转）扩散的最小 sigma 值。
            max_sigma (float): IGSO3（旋转）扩散的最大 sigma 值。
            min_b (float): IGSO3（旋转）扩散的最小 beta 值。
            max_b (float): IGSO3（旋转）扩散的最大 beta 值。
            schedule_type (str): 平移噪声调度表的类型 (例如 'linear')。
            so3_schedule_type (str): 旋转噪声调度表的类型 (例如 'linear')。
            so3_type (str): 旋转扩散的具体实现类型, 固定为 'igso3'。
            crd_scale (float): 坐标缩放因子，在加噪前对坐标进行归一化，以稳定扩散过程。
            schedule_kwargs (dict, optional): 噪声调度表生成的额外参数。默认为 {}。
            var_scale (float, optional): 平移噪声方差的缩放因子。默认为 1.0。
            cache_dir (str, optional): 用于存储预计算的 IGSO3 噪声调度表文件的目录。默认为 "."。
            partial_T (int, optional): 部分扩散的起始时间步。如果提供，则前向扩散只进行到此步骤。默认为 None。
            truncation_level (int, optional): IGSO3 预计算时使用的最大序列长度。默认为 2000。
        """
        self.T = T
        self.b_0 = b_0
        self.b_T = b_T
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.crd_scale = crd_scale
        self.var_scale = var_scale
        self.cache_dir = cache_dir
        self.device = device

        # get backbone frame diffuser
        self.so3_diffuser = IGSO3(
            T=self.T,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            schedule=so3_schedule_type,
            min_b=min_b,
            max_b=max_b,
            cache_dir=self.cache_dir,
            L=truncation_level,
            device=self.device,
        )

        # get backbone translation diffuser
        self.eucl_diffuser = EuclideanDiffuser(
            self.T, b_0, b_T, schedule_type=schedule_type, device=self.device, **schedule_kwargs
        )

        print("Successful diffuser __init__")



    def batch_diffuse_pose(
        self,
        xyz: torch.Tensor,
        seq: Optional[torch.Tensor],
        atom_mask: Optional[torch.Tensor],
        diffusion_mask: Optional[torch.Tensor],
        lengths: torch.Tensor,
        t_final: Optional[int] = None,
        include_motif_sidechains: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: 原子坐标，形状为 (N_total, 27, 3)。
            seq: 未使用，仅为与 ``diffuse_pose`` 接口对齐。
            atom_mask: 原子掩码，形状为 (N_total, 27)
            diffusion_mask: 残基级别冻结掩码，形状为 (N_total，)
                * True 表示保持 motif，不扩散。
                * False 表示该残基将被扩散。
            lengths: 每个样本的残基数，形状为 (B,)。
            t_final: 返回的最大时间步（1-indexed）
            include_motif_sidechains: 是否保留 motif 区域侧链。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - fa_stack: 形状为 (N_total, 27, 3) 的扩散轨迹。
                - xyz_true: 经过中心化的原始坐标，形状为 (N_total, 27, 3)。
        """
        device = xyz.device
        dtype = xyz.dtype

        lengths = lengths.to(device=device, dtype=torch.long)
        batch_size = lengths.numel()
        total_residues = int(lengths.sum().item())
        assert xyz.shape[0] == total_residues, "The first dimension of xyz must be the same as the number of residues"
        assert torch.sum(torch.isnan(xyz.squeeze()[:, :3]).any(dim=-1).any(dim=-1)) == 0    # check if any BB atoms are nan before centering


        # ======================= 中心化 =======================
        # 创建一个辅助索引，将每个残基映射到其所属的蛋白质样本 (0, 1, ..., B-1)，例如, 如果 lengths = [3, 2], batch_idx 将是 [0, 0, 0, 1, 1]
        batch_idx = torch.repeat_interleave(torch.arange(batch_size, device=device), lengths)
        if torch.sum(diffusion_mask) > 0:
            # Case 1: 存在Motif，基于每个蛋白质内部的Motif残基进行中心化
            
            # a. 仅提取所有motif残基的C-alpha原子坐标
            motif_ca_xyz = xyz[diffusion_mask, 1, :]
            # b. 找到这些motif残基对应的蛋白质索引
            motif_batch_idx = batch_idx[diffusion_mask]
            # c. 分组求和：使用index_add_高效计算每个蛋白质样本中motif坐标的总和
            motif_com_sum = torch.zeros(batch_size, 3, device=device, dtype=dtype)
            motif_com_sum.index_add_(0, motif_batch_idx, motif_ca_xyz)
            # d. 分组计数：计算每个蛋白质样本中有多少个motif残基
            num_motif_per_protein = torch.bincount(motif_batch_idx, minlength=batch_size).float().unsqueeze(1)
            # e. 分组求均值，并防止除以零
            coms_per_protein = motif_com_sum / num_motif_per_protein.clamp(min=1e-6)
            # f. 保存每个蛋白质的motif中心点 (现在是一个 [B, 3] 的张量)
            self.motif_com = coms_per_protein
            # g. 将每个蛋白质的中心点“广播”回其所有残基上
            coms_broadcasted = self.motif_com[batch_idx] # [N_total, 3]
            # h. 执行中心化：从每个残基坐标中减去其所属蛋白质的中心点
            xyz = xyz - coms_broadcasted.unsqueeze(1)
        else:
            # Case 2: 不存在Motif，基于每个蛋白质自身的全部残基进行中心化
            # a. 提取所有残基的C-alpha原子坐标
            ca_xyz = xyz[:, 1, :]
            # b. 分组求和
            coms_sum = torch.zeros(batch_size, 3, device=device, dtype=dtype)
            coms_sum.index_add_(0, batch_idx, ca_xyz)
            # c. 分组求均值
            coms_per_protein = coms_sum / lengths.unsqueeze(1).clamp(min=1e-6)
            # d. 广播并执行中心化
            coms_broadcasted = coms_per_protein[batch_idx]
            xyz = xyz - coms_broadcasted.unsqueeze(1)
        # ======================= 中心化 =======================
        xyz_true = torch.clone(xyz)
        xyz = xyz * self.crd_scale


        # get translations
        diffused_T, delta = self.eucl_diffuser.apply_kernel(
            x=xyz[:, :3, :].clone(),
            t=t_final,
            diffusion_mask=diffusion_mask
        )
        diffused_T /= self.crd_scale
        delta /= self.crd_scale


        # get frames
        diffused_frame_crds, diffused_frames = self.so3_diffuser.diffuse_frames(
            xyz[:, :3, :].clone(), final_t=t_final, diffusion_mask=diffusion_mask
        )
        diffused_frame_crds /= self.crd_scale


        # 最终的骨架坐标 = 仅旋转的坐标 + 总平移噪声。delta 形状为 (N_total, 3)，需要扩展维度以匹配 diffused_frame_crds
        diffused_BB = diffused_frame_crds + delta.unsqueeze(1)
        diffused_fa = torch.zeros(total_residues, 27, 3, device=device, dtype=dtype)
        diffused_fa[:, :3, :] = diffused_BB
        

        # if include_motif_sidechains and torch.any(diffusion_mask):  # 如果需要，添加 motif 区域的侧链 (从中心化但未缩放的真实坐标中获取)
        #     diffused_fa[diffusion_mask, :14, :] = xyz_true[diffusion_mask, :14, :]

        # $$这里进行修改：当include_motif_sidechains==true时，无论如何都添加所有残基的侧链
        if include_motif_sidechains:
            diffused_fa[:, 3:14, :] = xyz_true[:, 3:14, :]


        return diffused_fa, xyz_true
