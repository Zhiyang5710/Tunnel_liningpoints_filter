"""
TunnelPCSF - Cylindrical Cloth Simulation Filter (Vectorized)
================================================================
核心改进：
  1. 完全 NumPy 向量化：step() / apply_collision() 不再有任何 Python 循环
  2. KD-Tree 加速碰撞查询（scipy.spatial.cKDTree）
  3. 早停机制：连续多轮约束粒子数不变则提前结束
  4. 初始布料可视化：preview_cloth() 在模拟开始前输出图像供检查
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import time

# ─────────────────────────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────────────────────────

@dataclass
class TunnelPCSFParams:
    """
    Parameters for the Cylindrical Cloth Simulation Filter.

    cloth_resolution_angle : float  布料角度分辨率（度），越小越精细，默认 1.0
    cloth_resolution_z     : float  布料轴向分辨率（米），越小越精细，默认 0.1
    class_threshold        : float  衬砌分类径向阈值（米），默认 0.3
    rigidness              : int    布料刚性 1=软 2=中 3=硬，默认 2
    time_step              : float  Verlet 时间步长，默认 0.65
    iterations             : int    最大迭代次数，默认 500
    smooth_slope           : bool   是否对布料空洞做插值平滑，默认 True
    smooth_iterations      : int    平滑迭代次数，默认 5
    axis_method            : str    轴线估计方式 'pca' 或 'Slice' 或 'provided'
    initial_radius_offset  : float  布料初始半径比估计半径多出多少（米），默认 0.5
    early_stop_patience    : int    约束粒子数连续多少轮不变则提前停止，默认 10
    collision_search_angle : float  碰撞查询时每个粒子角度搜索范围（度），默认 2.0
    """
    cloth_resolution_angle : float = 1.0
    cloth_resolution_z     : float = 0.1
    class_threshold        : float = 0.1
    rigidness              : int   = 2
    time_step              : float = 0.5
    iterations             : int   = 200
    smooth_slope           : bool  = False
    smooth_iterations      : int   = 5
    axis_method            : str   = 'pca'
    initial_radius_offset  : float = 1.0
    early_stop_patience    : int   = 10
    collision_search_angle : float = 5.0   # degrees


# ─────────────────────────────────────────────────────────────
#  Vectorized Cylindrical Cloth
# ─────────────────────────────────────────────────────────────

class CylindricalCloth:
    """
    圆柱布料，全向量化实现。

    内部状态全部用 numpy 数组，形状均为 (n_z, n_theta, ...) 或展平后的版本。
    不再使用 Particle 对象，避免 Python 循环开销。

    关键数组：
      pos     : (n_z, n_theta, 3)  当前位置
      pos_old : (n_z, n_theta, 3)  上一时刻位置（Verlet 用）
      constrained : (n_z, n_theta) bool  是否被点云固定
      cloth_r     : (n_z, n_theta) float 每个粒子当前径向距离（缓存）
    """

    def __init__(
        self,
        axis_origin   : np.ndarray,
        axis_direction: np.ndarray,
        initial_radius: float,
        z_min: float,
        z_max: float,
        params: TunnelPCSFParams,
    ):
        self.axis_origin = axis_origin.astype(np.float64)
        self.axis_dir    = axis_direction.astype(np.float64)
        self.axis_dir   /= np.linalg.norm(self.axis_dir)
        self.params      = params

        self._build_axis_frame()

        self.n_theta = max(8, int(360.0 / params.cloth_resolution_angle))
        self.thetas  = np.linspace(0, 2 * np.pi, self.n_theta, endpoint=False)

        z_extent  = z_max - z_min
        self.n_z  = max(2, int(z_extent / params.cloth_resolution_z) + 1)
        self.z_values = np.linspace(z_min, z_max, self.n_z)

        self.initial_radius = initial_radius
        self.z_min = z_min
        self.z_max = z_max

        # ── 初始化粒子位置数组 ──────────────────────────────
        # 显式构造 (n_z, n_theta, 3)，避免广播歧义
        Z  = self.z_values          # (n_z,)
        TH = self.thetas            # (n_theta,)

        # 沿轴线方向的偏移: (n_z, 1, 3)
        axial  = Z[:, None, None] * self.axis_dir[None, None, :]   # (n_z, 1, 3)
        # 径向方向: (1, n_theta, 3)
        radial = initial_radius * (
            np.cos(TH)[None, :, None] * self.e1[None, None, :]
            + np.sin(TH)[None, :, None] * self.e2[None, None, :]
        )                                                           # (1, n_theta, 3)

        self.pos     = (self.axis_origin[None, None, :] + axial + radial).astype(np.float64)
        self.pos_old = self.pos.copy()

        # 约束标记
        self.constrained = np.zeros((self.n_z, self.n_theta), dtype=bool)

        # 计算初始径向距离（用于弹簧约束的 rest length）
        dz_rest = (z_max - z_min) / max(self.n_z - 1, 1)
        dt_rest = initial_radius * 2 * np.pi / self.n_theta
        self.rest_len_z = dz_rest    # 轴向相邻粒子静止长度
        self.rest_len_t = dt_rest    # 环向相邻粒子静止长度

    def _build_axis_frame(self):
        ax  = self.axis_dir
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ax, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        self.e1 = np.cross(ax, tmp); self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(ax, self.e1); self.e2 /= np.linalg.norm(self.e2)

    # ── 核心模拟步（完全向量化）──────────────────────────────

    def step(self):
        """
        一步 Verlet 积分 + 弹簧约束，无任何 Python for 循环。
        """
        dt = self.params.time_step

        # ── 1. 计算每个粒子的径向向内力方向 ──────────────────
        delta   = self.pos - self.axis_origin          # (nz, nt, 3)
        z_proj  = np.einsum('ijk,k->ij', delta, self.axis_dir)  # (nz, nt)
        radial  = delta - z_proj[:, :, None] * self.axis_dir    # (nz, nt, 3)
        r       = np.linalg.norm(radial, axis=2)                # (nz, nt)
        safe_r  = np.maximum(r, 1e-8)
        inward  = -radial / safe_r[:, :, None]                  # (nz, nt, 3)

        # ── 2. Verlet 积分（只对未固定粒子）──────────────────
        free  = ~self.constrained                               # (nz, nt) bool
        vel   = self.pos - self.pos_old                        # (nz, nt, 3)
        new_pos = self.pos + vel + inward * (dt * dt)

        self.pos_old = np.where(free[:, :, None], self.pos, self.pos_old)
        self.pos     = np.where(free[:, :, None], new_pos,  self.pos)

        # ── 3. 弹簧约束（环向 + 轴向），迭代 rigidness 次 ───
        for _ in range(self.params.rigidness):
            self._satisfy_constraints()

    def _satisfy_constraints(self):
        """
        向量化弹簧约束：同时处理所有环向对和轴向对。
        约束粒子不移动。
        """
        # ── 环向对 (it, it+1 mod n_theta) ──────────────────
        p1  = self.pos                                     # (nz, nt, 3)
        p2  = np.roll(self.pos, -1, axis=1)               # 环向下一粒子
        c1  = self.constrained
        c2  = np.roll(self.constrained, -1, axis=1)

        diff = p2 - p1
        dist = np.linalg.norm(diff, axis=2, keepdims=True).clip(1e-10)
        # 软约束：把偏离量对半均分给两端粒子
        correction = diff * 0.5

        # 只移动未固定的粒子
        move1 = (~c1 & ~c2)[:, :, None]
        move1_only = (~c1 & c2)[:, :, None]
        move2_only = (c1 & ~c2)[:, :, None]

        delta1 = np.where(move1, correction * 0.5,
                 np.where(move1_only, correction, np.zeros_like(correction)))
        delta2 = np.where(move1, -correction * 0.5,
                 np.where(move2_only, -correction, np.zeros_like(correction)))

        self.pos += delta1
        self.pos += np.roll(delta2, 1, axis=1)

        # ── 轴向对 (iz, iz+1) ───────────────────────────────
        p1  = self.pos[:-1]
        p2  = self.pos[1:]
        c1  = self.constrained[:-1]
        c2  = self.constrained[1:]

        diff = p2 - p1
        correction = diff * 0.5

        move1 = (~c1 & ~c2)[:, :, None]
        move1_only = (~c1 & c2)[:, :, None]
        move2_only = (c1 & ~c2)[:, :, None]

        d1 = np.where(move1, correction * 0.5,
             np.where(move1_only, correction, np.zeros_like(correction)))
        d2 = np.where(move1, -correction * 0.5,
             np.where(move2_only, -correction, np.zeros_like(correction)))

        self.pos[:-1] += d1
        self.pos[1:]  += d2

    def apply_collision(self, pts_cyl: np.ndarray):
        """
        向量化碰撞检测。

        思路：把点云按 (iz, it) 格子分桶，对每个格子取最大半径
        （最外层点云半径 = 衬砌面）；若布料粒子已经收缩到该半径内侧，
        则固定（pin）并 snap 到表面。

        pts_cyl : (N, 3) — (r, theta, z_along_axis)
        """
        n_z, n_t = self.n_z, self.n_theta
        theta_res = 2 * np.pi / n_t
        z_range   = self.z_max - self.z_min

        # ── 把每个点云点映射到最近的网格格子 ─────────────────
        it_arr = (pts_cyl[:, 1] / theta_res).astype(int) % n_t
        if z_range > 0:
            iz_raw = (pts_cyl[:, 2] - self.z_min) / z_range * (n_z - 1)
        else:
            iz_raw = np.zeros(len(pts_cyl))
        iz_arr = np.clip(iz_raw.astype(int), 0, n_z - 1)

        # ── 每个格子的最大点云半径（线性索引 trick）────────────
        flat_idx  = iz_arr * n_t + it_arr                    # (N,)
        cell_maxr = np.full(n_z * n_t, -np.inf)
        np.maximum.at(cell_maxr, flat_idx, pts_cyl[:, 0])
        cell_maxr = cell_maxr.reshape(n_z, n_t)              # (nz, nt)

        # ── 计算布料粒子当前半径 ───────────────────────────────
        delta  = self.pos - self.axis_origin
        z_proj = np.einsum('ijk,k->ij', delta, self.axis_dir)
        radial = delta - z_proj[:, :, None] * self.axis_dir
        cloth_r = np.linalg.norm(radial, axis=2)             # (nz, nt)

        # ── 判断是否碰撞 ───────────────────────────────────────
        has_point  = cell_maxr > -np.inf                     # 该格子有点云
        collide    = has_point & (~self.constrained) & (cloth_r <= cell_maxr)

        # ── Snap 碰撞粒子到点云表面 ───────────────────────────
        if collide.any():
            safe_r    = np.maximum(cloth_r, 1e-8)
            snap_r    = np.where(collide, cell_maxr, cloth_r) # (nz, nt)
            scale     = snap_r / safe_r                        # (nz, nt)
            # 只更新碰撞粒子的径向距离（保持轴向分量不变）
            snapped_pos = (self.axis_origin
                           + z_proj[:, :, None] * self.axis_dir
                           + radial * scale[:, :, None])
            self.pos         = np.where(collide[:, :, None], snapped_pos, self.pos)
            self.pos_old     = np.where(collide[:, :, None], snapped_pos, self.pos_old)
            self.constrained = self.constrained | collide

    def smooth_gaps(self):
        """
        向量化空洞平滑：对未固定粒子，用四邻域平均填补。
        """
        free = ~self.constrained
        if not free.any():
            return

        # 四邻域（环向wrap + 轴向clip）
        nb_sum = (
            np.roll(self.pos,  1, axis=1) +
            np.roll(self.pos, -1, axis=1) +
            np.concatenate([self.pos[:1], self.pos[:-1]], axis=0) +
            np.concatenate([self.pos[1:], self.pos[-1:]], axis=0)
        )
        nb_avg = nb_sum / 4.0

        # 统计邻居中有多少被固定（固定邻居权重更高）
        c_roll = np.stack([
            np.roll(self.constrained,  1, axis=1),
            np.roll(self.constrained, -1, axis=1),
            np.concatenate([self.constrained[:1], self.constrained[:-1]], axis=0),
            np.concatenate([self.constrained[1:], self.constrained[-1:]], axis=0),
        ], axis=0)                              # (4, nz, nt)
        n_constrained_nb = c_roll.sum(axis=0)  # (nz, nt)

        # 只对 ≥2 个固定邻居的自由粒子做平滑
        do_smooth = free & (n_constrained_nb >= 2)
        self.pos = np.where(do_smooth[:, :, None], nb_avg, self.pos)

    def get_cloth_radius_grid(self) -> np.ndarray:
        """
        返回布料每个粒子的当前径向距离 (n_z, n_theta)。
        """
        delta  = self.pos - self.axis_origin
        z_proj = np.einsum('ijk,k->ij', delta, self.axis_dir)
        radial = delta - z_proj[:, :, None] * self.axis_dir
        return np.linalg.norm(radial, axis=2)

    def get_cloth_radius_at_batch(
        self, theta: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """
        双线性插值，批量查询布料半径。

        theta : (N,) 弧度
        z     : (N,) 轴向坐标

        Returns : (N,) 对应布料半径
        """
        cloth_r = self.get_cloth_radius_grid()   # (nz, nt)
        n_z, n_t = self.n_z, self.n_theta
        theta_res = 2 * np.pi / n_t

        # θ 方向
        it_f = theta / theta_res
        it0  = np.floor(it_f).astype(int) % n_t
        it1  = (it0 + 1) % n_t
        wt   = it_f - np.floor(it_f)

        # z 方向
        z_range = self.z_max - self.z_min
        if z_range > 0:
            iz_f = (z - self.z_min) / z_range * (n_z - 1)
        else:
            iz_f = np.zeros(len(z))
        iz0  = np.clip(np.floor(iz_f).astype(int), 0, n_z - 1)
        iz1  = np.clip(iz0 + 1,                    0, n_z - 1)
        wz   = np.clip(iz_f - np.floor(iz_f), 0, 1)

        r00 = cloth_r[iz0, it0]
        r01 = cloth_r[iz0, it1]
        r10 = cloth_r[iz1, it0]
        r11 = cloth_r[iz1, it1]

        return ((1 - wz) * ((1 - wt) * r00 + wt * r01) +
                wz       * ((1 - wt) * r10 + wt * r11))

    def n_constrained_count(self) -> int:
        return int(self.constrained.sum())

    def get_initial_cloth_points(self) -> np.ndarray:
        """
        返回布料当前所有粒子的 XYZ 坐标，形状 (n_z*n_theta, 3)。
        用于可视化。
        """
        return self.pos.reshape(-1, 3)


# ─────────────────────────────────────────────────────────────
#  轴线估计（保持不变，引用自 run_tunnel_csf.py）
# ─────────────────────────────────────────────────────────────

def estimate_axis_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    sample   = points[np.random.choice(len(points), min(len(points), 100_000), replace=False)]
    cov      = np.cov((sample - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return centroid, eigvecs[:, np.argmax(eigvals)]


def estimate_axis_slice_centers(
    points: np.ndarray, n_slices: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    centroid, coarse_dir = estimate_axis_pca(points)
    coarse_dir /= np.linalg.norm(coarse_dir)

    tmp = np.array([1.,0.,0.]) if abs(np.dot(coarse_dir,[1,0,0])) < 0.9 else np.array([0.,1.,0.])
    e1  = np.cross(coarse_dir, tmp); e1 /= np.linalg.norm(e1)
    e2  = np.cross(coarse_dir, e1);  e2 /= np.linalg.norm(e2)

    delta  = points - centroid
    z_proj = delta @ coarse_dir
    z_min, z_max = z_proj.min(), z_proj.max()
    edges  = np.linspace(z_min, z_max, n_slices + 1)

    centers = []
    for i in range(n_slices):
        mask   = (z_proj >= edges[i]) & (z_proj < edges[i+1])
        if mask.sum() < 10: continue
        s  = delta[mask]
        u  = s @ e1;  v = s @ e2
        A  = np.column_stack([2*u, 2*v, np.ones(len(u))])
        b  = u**2 + v**2
        res, *_ = np.linalg.lstsq(A, b, rcond=None)
        z_mid = (edges[i] + edges[i+1]) / 2
        centers.append(centroid + z_mid*coarse_dir + res[0]*e1 + res[1]*e2)

    if len(centers) < 3:
        return estimate_axis_pca(points)

    centers = np.array(centers)
    cc = np.mean(centers, axis=0)
    _, _, Vt = np.linalg.svd(centers - cc)
    return cc, Vt[0]


def auto_estimate_axis(
    points: np.ndarray, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    def score(o, d, pts):
        delta  = pts - o
        z_proj = delta @ d
        radial = delta - z_proj[:, None] * d
        r      = np.linalg.norm(radial, axis=1)
        mask   = r > np.median(r) * 0.5
        return np.std(r[mask]), np.median(r)

    o1, d1 = estimate_axis_pca(points)
    d1 /= np.linalg.norm(d1)
    s1, r1 = score(o1, d1, points)

    o2, d2 = estimate_axis_slice_centers(points)
    d2 /= np.linalg.norm(d2)
    s2, r2 = score(o2, d2, points)

    if s1 <= s2:
        best_o, best_d, best_r, method = o1, d1, r1, 'pca'
    else:
        best_o, best_d, best_r, method = o2, d2, r2, 'slice'

    if verbose:
        print(f"  PCA   score={s1:.4f}m, r={r1:.3f}m")
        print(f"  Slice score={s2:.4f}m, r={r2:.3f}m")
        print(f"  → 选择 {method.upper()}, 估计半径={best_r:.3f}m")
        ang = np.degrees(np.arccos(np.clip(abs(np.dot(best_d,[1,0,0])),0,1)))
        print(f"  轴线与X夹角 {ang:.1f}°, 与Y {np.degrees(np.arccos(np.clip(abs(np.dot(best_d,[0,1,0])),0,1))):.1f}°, 与Z {np.degrees(np.arccos(np.clip(abs(np.dot(best_d,[0,0,1])),0,1))):.1f}°")

    return best_o, best_d, float(best_r), method


# ─────────────────────────────────────────────────────────────
#  可视化模块（问题2：初始布料预览）
# ─────────────────────────────────────────────────────────────

def preview_cloth(
    points        : np.ndarray,
    cloth         : 'CylindricalCloth',
    axis_origin   : np.ndarray,
    axis_dir      : np.ndarray,
    save_path     : str = "cloth_preview.png",
    subsample_pts : int = 20_000,
    show          : bool = True,
):
    """
    在模拟开始前，将初始布料网格与点云一起可视化。

    生成 3 个子图：
      左：XY 平面投影（截面视图）
      中：沿轴线展开的 θ-z 展开图（检查布料网格密度）
      右：3D 透视图（随机采样点 + 布料环线）

    Parameters
    ----------
    points       : 原始点云 (N,3)
    cloth        : 已初始化的 CylindricalCloth 对象
    axis_origin  : 隧道轴线原点
    axis_dir     : 隧道轴线方向（单位向量）
    save_path    : 输出图片路径
    subsample_pts: 可视化用的点云采样数（太多会慢）
    show         : 是否调用 plt.show()（在有界面时使用）

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    try:
        import matplotlib
        matplotlib.use('Agg')   # 无界面环境也能保存
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError:
        print("❌ 请先安装 matplotlib：pip install matplotlib")
        return None

    # ── 构建坐标系 ────────────────────────────────────────────
    d   = axis_dir / np.linalg.norm(axis_dir)
    tmp = np.array([1.,0.,0.]) if abs(np.dot(d,[1,0,0])) < 0.9 else np.array([0.,1.,0.])
    e1  = np.cross(d, tmp); e1 /= np.linalg.norm(e1)
    e2  = np.cross(d, e1);  e2 /= np.linalg.norm(e2)

    # ── 把点云投影到圆柱坐标 ──────────────────────────────────
    delta   = points - axis_origin
    z_proj  = delta @ d
    radial  = delta - z_proj[:,None] * d
    r_pts   = np.linalg.norm(radial, axis=1)
    cos_t   = (radial @ e1) / np.maximum(r_pts, 1e-10)
    sin_t   = (radial @ e2) / np.maximum(r_pts, 1e-10)
    theta_pts = np.arctan2(sin_t, cos_t)          # (-π, π)

    # ── 子采样点云 ─────────────────────────────────────────────
    n_show = min(subsample_pts, len(points))
    idx    = np.random.choice(len(points), n_show, replace=False)
    pts_show  = points[idx]
    r_show    = r_pts[idx]
    th_show   = theta_pts[idx]
    z_show    = z_proj[idx]

    # ── 布料粒子的圆柱坐标 ────────────────────────────────────
    cloth_pos = cloth.pos   # (nz, nt, 3)
    nz, nt, _ = cloth_pos.shape

    delta_c  = cloth_pos - axis_origin
    zc_proj  = np.einsum('ijk,k->ij', delta_c, d)          # (nz, nt)
    rc_rad   = delta_c - zc_proj[:,:,None] * d              # (nz, nt, 3)
    rc_r     = np.linalg.norm(rc_rad, axis=2)               # (nz, nt)
    rc_e1    = np.einsum('ijk,k->ij', rc_rad, e1) / np.maximum(rc_r, 1e-10)
    rc_e2    = np.einsum('ijk,k->ij', rc_rad, e2) / np.maximum(rc_r, 1e-10)
    rc_theta = np.arctan2(rc_e2, rc_e1)                     # (nz, nt)

    # ── 布置图形 ───────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6), facecolor='#1a1a2e')
    fig.suptitle('TunnelPCSF — Initial Cloth Preview(Before running)',
                 color='white', fontsize=14, y=0.98)

    ax_style = dict(facecolor='#16213e')
    txt_kw   = dict(color='#e0e0e0', fontsize=9)

    # ════════════════════════════════════════════════════
    #  子图1：截面视图（XY 平面，取中间 10% 点云）
    # ════════════════════════════════════════════════════
    ax1 = fig.add_subplot(131, **ax_style)
    ax1.set_facecolor('#16213e')
    ax1.tick_params(colors='#aaaaaa', labelsize=8)
    for sp in ax1.spines.values(): sp.set_color('#444466')

    # 只取中间段的点（轴向范围中间20%）
    z_mid  = (z_proj.min() + z_proj.max()) / 2
    z_half = (z_proj.max() - z_proj.min()) * 0.1
    mask_mid = np.abs(z_proj[idx] - z_mid) < z_half
    xs_pts = (radial[idx][mask_mid] @ e1)
    ys_pts = (radial[idx][mask_mid] @ e2)

    ax1.scatter(xs_pts, ys_pts, s=0.8, c='#4fc3f7', alpha=0.5,
                rasterized=True, label='Point Cloud (Middle section)')

    # 布料中间环（iz=nz//2）
    iz_mid = nz // 2
    cloth_ring = cloth_pos[iz_mid]                           # (nt, 3)
    ring_x = (cloth_ring - axis_origin) @ e1
    ring_y = (cloth_ring - axis_origin) @ e2
    # 取径向分量（去掉轴向）
    ring_ax  = np.einsum('ij,j->i', cloth_ring - axis_origin, d)
    ring_rad = (cloth_ring - axis_origin) - ring_ax[:,None] * d
    ring_x   = ring_rad @ e1
    ring_y   = ring_rad @ e2

    ax1.plot(np.append(ring_x, ring_x[0]),
             np.append(ring_y, ring_y[0]),
             'o-', color='#ffd54f', lw=1.5, ms=2,
             label=f'Initial cloth r={cloth.initial_radius:.2f}m')

    # 轴线
    ax1.plot(0, 0, '+', color='#ef5350', ms=4, mew=2, label='Tunnel Axis')

    ax1.set_aspect('equal')
    ax1.set_title('Section view (Middle section)', **txt_kw)
    ax1.set_xlabel('e₁ (m)', color='#aaaaaa', fontsize=8)
    ax1.set_ylabel('e₂ (m)', color='#aaaaaa', fontsize=8)
    ax1.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
               loc='upper right', markerscale=3)

    # 标注半径
    r_max_pts = r_pts.max()
    r_median  = np.median(r_pts)
    ax1.add_patch(plt.Circle((0,0), cloth.initial_radius,
                              color='#ffd54f', fill=False, lw=1, ls='--', alpha=0.5))
    ax1.add_patch(plt.Circle((0,0), r_median,
                              color='#81c784', fill=False, lw=1, ls=':', alpha=0.7))
    ax1.text(cloth.initial_radius * 0.7, cloth.initial_radius * 1.05,
             f'Cloth initialize r={cloth.initial_radius:.2f}m', color='#ffd54f', fontsize=7)
    ax1.text(r_median * 0.7, r_median * 1.05,
             f'Median of Point cloud (Dianyunzhongwei) r={r_median:.2f}m', color='#81c784', fontsize=7)

    # ════════════════════════════════════════════════════
    #  子图2：θ-z 展开图
    # ════════════════════════════════════════════════════
    ax2 = fig.add_subplot(132, **ax_style)
    ax2.set_facecolor('#16213e')
    ax2.tick_params(colors='#aaaaaa', labelsize=8)
    for sp in ax2.spines.values(): sp.set_color('#444466')

    # 点云按半径着色
    sc = ax2.scatter(
        np.degrees(th_show), z_show,
        c=r_show, cmap='cool', s=0.5, alpha=0.4,
        vmin=r_pts.min(), vmax=r_pts.max(), rasterized=True
    )
    cbar = fig.colorbar(sc, ax=ax2, pad=0.02)
    cbar.ax.tick_params(colors='#aaaaaa', labelsize=7)
    cbar.set_label('Radial distance (Jingxiangjuli) r (m)', color='#aaaaaa', fontsize=8)

    # 布料网格点（θ-z 投影）
    ax2.scatter(
        np.degrees(rc_theta.ravel()), zc_proj.ravel(),
        s=2, c='#ffd54f', alpha=0.6, zorder=5, label='Cloth Particle'
    )

    ax2.set_title('θ-z unfolding diagram (Cloth Mesh density check)', **txt_kw)
    ax2.set_xlabel('Angle θ (°)', color='#aaaaaa', fontsize=8)
    ax2.set_ylabel('Axial Position z_(m)', color='#aaaaaa', fontsize=8)
    ax2.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', markerscale=3)
    ax2.set_xlim(-180, 180)

    # 标注布料参数
    ax2.text(0.02, 0.98,
             f'Cloth SIZE: {nt}(θ) × {nz}(z)\n'
             f'Angular resolution: {360/nt:.1f}°\n'
             f'Axial resolution: {(cloth.z_max-cloth.z_min)/max(nz-1,1):.3f}m\n'
             f'Partical number: {nz*nt:,}',
             transform=ax2.transAxes,
             va='top', ha='left',
             color='#ffd54f', fontsize=8,
             bbox=dict(facecolor='#1a1a2e', alpha=0.8, edgecolor='none'))

    # ════════════════════════════════════════════════════
    #  子图3：3D 透视图
    # ════════════════════════════════════════════════════
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_facecolor('#16213e')
    ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
    ax3.xaxis.pane.set_edgecolor('#333355')
    ax3.yaxis.pane.set_edgecolor('#333355')
    ax3.zaxis.pane.set_edgecolor('#333355')
    ax3.tick_params(colors='#888888', labelsize=7)

    # 点云（子采样）
    n3d = min(8000, len(pts_show))
    idx3 = np.random.choice(len(pts_show), n3d, replace=False)
    ax3.scatter(pts_show[idx3, 0], pts_show[idx3, 1], pts_show[idx3, 2],
                s=0.5, c='#4fc3f7', alpha=0.3, rasterized=True)

    # 布料环线（每隔几个 iz 画一条）
    step_iz = max(1, nz // 10)
    for iz in range(0, nz, step_iz):
        ring = cloth_pos[iz]                                 # (nt, 3)
        ring = np.vstack([ring, ring[0]])                    # 闭合
        ax3.plot(ring[:,0], ring[:,1], ring[:,2],
                 '-', color='#ffd54f', lw=0.8, alpha=0.7)

    # 布料轴线方向的纵线（每隔几个 it 画一条）
    step_it = max(1, nt // 16)
    for it in range(0, nt, step_it):
        line = cloth_pos[:, it, :]                           # (nz, 3)
        ax3.plot(line[:,0], line[:,1], line[:,2],
                 '-', color='#ffd54f', lw=0.5, alpha=0.4)

    # 隧道轴线
    ax_len = (cloth.z_max - cloth.z_min) * 0.5
    ax_pts = np.array([
        axis_origin + t * d for t in np.linspace(-ax_len*0.1, ax_len*1.1, 20)
    ])
    ax3.plot(ax_pts[:,0], ax_pts[:,1], ax_pts[:,2],
             '--', color='#ef5350', lw=1.5, alpha=0.8, label='Tunnel Axis')

    ax3.set_title('3D perpective view', **txt_kw)
    ax3.set_xlabel('X', color='#888888', fontsize=7)
    ax3.set_ylabel('Y', color='#888888', fontsize=7)
    ax3.set_zlabel('Z', color='#888888', fontsize=7)

    # 信息框
    info = (
        f"Point cloud: {len(points):,} pts\n"
        f"Initial radius: {cloth.initial_radius:.3f} m\n"
        f"Median radius of Point Cloud: {np.median(r_pts):.3f} m\n"
        f"Axial range: {cloth.z_min:.2f}~{cloth.z_max:.2f} m\n"
        f"Cloth particle: {nz*nt:,}"
    )
    ax3.text2D(0.02, 0.98, info, transform=ax3.transAxes,
               va='top', color='#e0e0e0', fontsize=7,
               bbox=dict(facecolor='#1a1a2e', alpha=0.85, edgecolor='none'))

    plt.tight_layout(rect=[0,0,1,0.96])

    # 保存
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"[预览] 初始布料图已保存至: {save_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass

    return fig


# ─────────────────────────────────────────────────────────────
#  主过滤器类
# ─────────────────────────────────────────────────────────────

class TunnelPCSF:
    """
    圆柱布料模拟隧道点云滤波器（向量化版本）。

    用法
    ----
    csf = TunnelPCSF()
    csf.params.class_threshold = 0.2
    csf.set_point_cloud(points)

    # 可选：模拟前先检查初始布料
    csf.preview_initial_cloth(save_path='check.png')

    lining_idx, interior_idx = csf.do_filtering()
    """

    def __init__(self):
        self.params        = TunnelPCSFParams()
        self._points       : Optional[np.ndarray] = None
        self._axis_origin  : Optional[np.ndarray] = None
        self._axis_dir     : Optional[np.ndarray] = None
        self._pts_cyl      : Optional[np.ndarray] = None
        self._cloth        : Optional[CylindricalCloth] = None

    def set_point_cloud(self, points: np.ndarray):
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points must be (N, 3)")
        self._points  = points[:, :3].astype(np.float64)
        self._pts_cyl = None
        self._cloth   = None

    def set_axis(self, origin: np.ndarray, direction: np.ndarray):
        self._axis_origin = np.array(origin,    dtype=np.float64)
        self._axis_dir    = np.array(direction, dtype=np.float64)
        self._axis_dir   /= np.linalg.norm(self._axis_dir)
        self.params.axis_method = 'provided'

    def _ensure_axis_and_cyl(self, verbose: bool):
        """估计轴线 + 转换圆柱坐标（幂等，多次调用只算一次）。"""
        if self._axis_origin is None or self.params.axis_method == 'pca':
            if verbose: print("[TunnelPCSF] 估计隧道轴线...")
            self._axis_origin, self._axis_dir, _, _ = auto_estimate_axis(
                self._points, verbose=verbose
            )
        if self._pts_cyl is None:
            if verbose: print("[TunnelPCSF] 转换圆柱坐标...")
            self._pts_cyl = self._to_cylindrical_batch(self._points)

    def _build_cloth(self, verbose: bool) -> CylindricalCloth:
        """构建圆柱布料对象。"""
        pts_cyl = self._pts_cyl
        z_min, z_max  = pts_cyl[:, 2].min(), pts_cyl[:, 2].max()
        median_r      = float(np.median(pts_cyl[:, 0]))
        # initial_radius = median_r + self.params.initial_radius_offset
        initial_radius = median_r * 1.5

        cloth = CylindricalCloth(
            axis_origin   = self._axis_origin,
            axis_direction= self._axis_dir,
            initial_radius= initial_radius,
            z_min=z_min, z_max=z_max,
            params=self.params,
        )
        if verbose:
            print(f"  估计半径: {median_r:.3f}m  布料初始半径(1.5倍扩大): {initial_radius:.3f}m")
            print(f"  布料网格: {cloth.n_theta}(θ) × {cloth.n_z}(z) = {cloth.n_theta*cloth.n_z:,} 粒子")
        return cloth

    def preview_initial_cloth(
        self,
        save_path     : str  = "cloth_preview.png",
        subsample_pts : int  = 20_000,
        show          : bool = False,
    ):
        """
        在运行模拟之前，可视化初始布料与点云的关系。
        用于检查初始半径、布料密度是否合适。

        Parameters
        ----------
        save_path     : 图片保存路径（默认 cloth_preview.png）
        subsample_pts : 可视化用的点云采样数量
        show          : 是否弹出交互窗口（服务器环境设 False）
        """
        if self._points is None:
            raise RuntimeError("请先调用 set_point_cloud()")

        self._ensure_axis_and_cyl(verbose=True)

        if self._cloth is None:
            self._cloth = self._build_cloth(verbose=True)

        return preview_cloth(
            points       = self._points,
            cloth        = self._cloth,
            axis_origin  = self._axis_origin,
            axis_dir     = self._axis_dir,
            save_path    = save_path,
            subsample_pts= subsample_pts,
            show         = show,
        )

    def do_filtering(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行完整的圆柱布料模拟滤波流程。

        Returns
        -------
        lining_idx   : 衬砌点索引
        interior_idx : 非衬砌点索引
        """
        if self._points is None:
            raise RuntimeError("请先调用 set_point_cloud()")

        t0 = time.time()

        # ── 1. 轴线 + 圆柱坐标 ────────────────────────────
        self._ensure_axis_and_cyl(verbose)
        pts_cyl = self._pts_cyl

        # ── 2. 初始化布料（若未预览则新建）────────────────
        if self._cloth is None:
            if verbose: print("[TunnelPCSF] 初始化布料...")
            self._cloth = self._build_cloth(verbose)
        cloth = self._cloth

        # ── 3. 模拟迭代 ────────────────────────────────────
        if verbose:
            print(f"[TunnelPCSF] 开始模拟（最多 {self.params.iterations} 轮）...")

        patience    = self.params.early_stop_patience
        prev_count  = -1
        stable_cnt  = 0

        for i in range(self.params.iterations):
            t_iter = time.time()
            cloth.step()
            cloth.apply_collision(pts_cyl)

            n_pin = cloth.n_constrained_count()

            # 早停检查
            if n_pin == prev_count:
                stable_cnt += 1
            else:
                stable_cnt = 0
            prev_count = n_pin

            if verbose and ((i + 1) % 50 == 0 or stable_cnt >= patience):
                total = cloth.n_theta * cloth.n_z
                print(f"  Iter {i+1:4d} | 已固定 {n_pin}/{total} "
                      f"({100*n_pin/total:.1f}%) | {time.time()-t_iter:.3f}s/iter")

            if stable_cnt >= patience:
                if verbose:
                    print(f"  ✓ 早停：连续 {patience} 轮无新固定粒子，提前结束")
                break

        # ── 4. 后处理平滑 ──────────────────────────────────
        if self.params.smooth_slope:
            if verbose: print("[TunnelPCSF] 后处理平滑...")
            for _ in range(self.params.smooth_iterations):
                cloth.smooth_gaps()

        # ── 5. 向量化分类 ──────────────────────────────────
        if verbose: print("[TunnelPCSF] 分类点云...")

        r_cloth = cloth.get_cloth_radius_at_batch(pts_cyl[:, 1], pts_cyl[:, 2])
        dist    = r_cloth - pts_cyl[:, 0]       # > 0: 点在布料内侧

        threshold    = self.params.class_threshold
        lining_mask  = np.abs(dist) <= threshold
        # 额外：在布料外侧但距离很近的点也归为衬砌（snap 残差）
        lining_mask |= (dist < 0) & (np.abs(dist) <= threshold * 1.5)

        lining_idx   = np.where(lining_mask)[0].astype(np.int64)
        interior_idx = np.where(~lining_mask)[0].astype(np.int64)

        N = len(self._points)
        if verbose:
            print(f"[TunnelPCSF] 完成！总耗时 {time.time()-t0:.1f}s")
            print(f"  衬砌点  : {len(lining_idx):,} ({100*len(lining_idx)/N:.1f}%)")
            print(f"  非衬砌点: {len(interior_idx):,} ({100*len(interior_idx)/N:.1f}%)")

        return lining_idx, interior_idx

    def save_points(self, indices: np.ndarray, filepath: str):
        if self._points is None:
            raise RuntimeError("No point cloud loaded.")
        np.savetxt(filepath, self._points[indices], fmt='%.6f')
        print(f"[TunnelPCSF] 已保存 {len(indices):,} 点到 {filepath}")

    def _to_cylindrical_batch(self, points: np.ndarray) -> np.ndarray:
        o, d = self._axis_origin, self._axis_dir
        tmp  = np.array([1.,0.,0.]) if abs(np.dot(d,[1,0,0])) < 0.9 else np.array([0.,1.,0.])
        e1   = np.cross(d, tmp); e1 /= np.linalg.norm(e1)
        e2   = np.cross(d, e1);  e2 /= np.linalg.norm(e2)

        delta = points - o
        z     = delta @ d
        radial= delta - z[:, None] * d
        r     = np.linalg.norm(radial, axis=1)
        cos_t = (radial @ e1) / np.maximum(r, 1e-10)
        sin_t = (radial @ e2) / np.maximum(r, 1e-10)
        theta = np.arctan2(sin_t, cos_t) % (2 * np.pi)

        return np.stack([r, theta, z], axis=1)
