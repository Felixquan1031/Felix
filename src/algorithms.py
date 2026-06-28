"""
算法模块（v3.0）：
- DC-HQS-AGST：基于 DC 分解、HQS 变量分裂与 FFT 闭式求解的高效算法
- DC-HQS-AGST-L0：在 LN 基础上进一步引入 L0 梯度先验的混合模型
- 基准方法：FISTA-LASSO、FISTA-Huber、PG-Log-L1（从 v2.0 保留）
"""

import numpy as np
from .operators import (
    soft_threshold, gst, agst, hard_threshold_grad,
    DCLogDecomposition, GradientOperator,
    objective_log, compute_psnr,
)


# ==================== 基准方法（保留 v2.0 实现） ====================

def fista_lasso(x0, A, b, lam, L=None, max_iter=300, tol=1e-6, verbose=False):
    if L is None:
        L = A.lipschitz
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    obj_history = []
    for k in range(max_iter):
        grad = A.adjoint(A.residual(y, b))
        x_new = soft_threshold(y - grad / L, lam / L)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
        y_new = x_new + ((t - 1.0) / t_new) * (x_new - x)
        obj = 0.5 * np.sum(A.residual(x_new, b)**2) + lam * np.sum(np.abs(x_new))
        obj_history.append(obj)
        if k > 0 and obj > obj_history[-2]:
            t_new = 1.0
            y_new = x_new.copy()
        rel_change = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-12)
        if verbose and k % 50 == 0:
            print(f"[FISTA-LASSO] iter {k:4d}, obj={obj:.6e}, rel_change={rel_change:.3e}")
        if rel_change < tol:
            break
        x, y, t = x_new, y_new, t_new
    return x, obj_history


def fista_huber(x0, A, b, lam, c=1.0, L=None, max_iter=300, tol=1e-6, verbose=False):
    if L is None:
        L = A.lipschitz
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    obj_history = []
    for k in range(max_iter):
        r = A.residual(y, b)
        grad = A.adjoint(np.clip(r, -c, c))
        x_new = soft_threshold(y - grad / L, lam / L)
        abs_r = np.abs(A.residual(x_new, b))
        huber_loss = np.where(abs_r <= c, 0.5 * abs_r**2, c * abs_r - 0.5 * c**2)
        obj = np.sum(huber_loss) + lam * np.sum(np.abs(x_new))
        obj_history.append(obj)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
        y_new = x_new + ((t - 1.0) / t_new) * (x_new - x)
        if k > 0 and obj > obj_history[-2]:
            t_new = 1.0
            y_new = x_new.copy()
        rel_change = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-12)
        if verbose and k % 50 == 0:
            print(f"[FISTA-Huber] iter {k:4d}, obj={obj:.6e}, rel_change={rel_change:.3e}")
        if rel_change < tol:
            break
        x, y, t = x_new, y_new, t_new
    return x, obj_history


def proximal_gradient_log(x0, A, b, lam, step=None, max_iter=800, tol=1e-6, verbose=False):
    if step is None:
        step = 0.5 / A.lipschitz
    x = x0.copy()
    obj_history = []
    for k in range(max_iter):
        r = A.residual(x, b)
        grad = A.adjoint(2.0 * r / (1.0 + r**2))
        x_new = soft_threshold(x - step * grad, lam * step)
        obj = np.sum(np.log1p(A.residual(x_new, b)**2)) + lam * np.sum(np.abs(x_new))
        obj_history.append(obj)
        rel_change = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-12)
        if verbose and k % 100 == 0:
            print(f"[PG-Log-L1] iter {k:4d}, obj={obj:.6e}, rel_change={rel_change:.3e}")
        if rel_change < tol:
            break
        x = x_new
    return x, obj_history


# ==================== 公平对比：梯度域 HQS-L1 基准 ====================

def hqs_l2_l1(b, kernel, A, lam,
              mu0=0.01, mu_max=10.0, n_hqs=10,
              verbose=False, return_history=True):
    """
    梯度域 LASSO 的 HQS 求解：
        min_x 0.5*||Ax-b||^2 + lambda*||∇x||_1
    作为与 DC-HQS-AGST 的公平基准（同域、同 FFT 求解框架）。
    """
    m, n = b.shape
    x = b.copy()
    grad_op = GradientOperator((m, n))

    k_fft = np.fft.fft2(kernel, s=(m, n))
    k_conj = np.conj(k_fft)
    k_sq = np.abs(k_fft)**2
    grad_sq = grad_op.grad_sq_fft

    b_fft = np.fft.fft2(b)
    mu = mu0
    obj_history = []
    outer_iter = 0

    while mu <= mu_max:
        if verbose:
            print(f"\n[HQS-L2-L1] μ={mu:.4f}, outer_iter={outer_iter}")
        for hqs_iter in range(n_hqs):
            dh, dv = grad_op.forward(x)
            # u-update: 软阈值（L1 近端）
            u_h = soft_threshold(dh, lam / mu)
            u_v = soft_threshold(dv, lam / mu)

            # x-update: FFT 闭式
            u_h_fft = np.fft.fft2(u_h)
            u_v_fft = np.fft.fft2(u_v)
            grad_term = (np.conj(grad_op.grad_h_fft) * u_h_fft +
                         np.conj(grad_op.grad_v_fft) * u_v_fft)
            numerator = k_conj * b_fft + mu * grad_term
            denominator = k_sq + mu * grad_sq
            x_fft = numerator / denominator
            x = np.real(np.fft.ifft2(x_fft))
            x = np.clip(x, 0.0, 1.0)

            if return_history:
                dh, dv = grad_op.forward(x)
                obj = (0.5 * np.sum(A.residual(x, b)**2) +
                       lam * (np.sum(np.abs(dh)) + np.sum(np.abs(dv))))
                obj_history.append(obj)
            if verbose and hqs_iter % 2 == 0:
                print(f"  HQS[{hqs_iter}] obj={obj_history[-1]:.6e}")
        mu *= 2.0
        outer_iter += 1

    return x, obj_history


def hqs_huber_l1(b, kernel, A, lam, c,
                 mu0=0.01, mu_max=10.0, n_hqs=10, n_pg=5,
                 verbose=False, return_history=True):
    """
    梯度域 Huber-LASSO 的 HQS 求解：
        min_x sum Huber(Ax-b, c) + lambda*||∇x||_1
    x 子问题通过 PG 近似求解（Huber 损失无 FFT 闭式解）。
    """
    m, n = b.shape
    x = b.copy()
    grad_op = GradientOperator((m, n))

    k_fft = np.fft.fft2(kernel, s=(m, n))
    k_sq = np.abs(k_fft)**2
    grad_sq = grad_op.grad_sq_fft

    mu = mu0
    obj_history = []
    outer_iter = 0

    while mu <= mu_max:
        if verbose:
            print(f"\n[HQS-Huber-L1] μ={mu:.4f}, outer_iter={outer_iter}")
        for hqs_iter in range(n_hqs):
            dh, dv = grad_op.forward(x)
            # u-update: 软阈值
            u_h = soft_threshold(dh, lam / mu)
            u_v = soft_threshold(dv, lam / mu)

            # x-update: 对 Huber + 二次惩罚做若干步 PG
            step = 1.0 / (A.lipschitz + 8.0 * mu)
            for _ in range(n_pg):
                r = A.residual(x, b)
                grad_data = A.adjoint(np.clip(r, -c, c))
                grad_reg = mu * (grad_op.adjoint(*grad_op.forward(x)) -
                                 grad_op.adjoint(u_h, u_v))
                x = x - step * (grad_data + grad_reg)
                x = np.clip(x, 0.0, 1.0)

            if return_history:
                r = A.residual(x, b)
                abs_r = np.abs(r)
                huber_loss = np.where(abs_r <= c, 0.5 * abs_r**2,
                                      c * abs_r - 0.5 * c**2)
                dh, dv = grad_op.forward(x)
                obj = np.sum(huber_loss) + lam * (np.sum(np.abs(dh)) + np.sum(np.abs(dv)))
                obj_history.append(obj)
            if verbose and hqs_iter % 2 == 0:
                print(f"  HQS[{hqs_iter}] obj={obj_history[-1]:.6e}")
        mu *= 2.0
        outer_iter += 1

    return x, obj_history


# ==================== 改进方法：DC-HQS-AGST ====================

def dc_hqs_agst(b, kernel, A, lam, p=0.8,
                mu0=0.01, mu_max=10.0,
                n_dc=2, n_hqs=5,
                verbose=False, return_history=True):
    """
    DC-HQS-AGST 算法（核心改进）。

    模型：min_x  sum log(1+(Ax-b)^2) + lam * ||∇x||_p^p / ||∇x||_inf

    算法流程：
    1. DC 外循环：将对数损失分解为 L2 - h(r)，构造修正右端项 b̃。
    2. HQS 内循环：引入辅助变量 u = ∇x，交替更新 u（AGST）与 x（FFT 闭式）。
    3. 惩罚参数递增：μ ← 2μ，直至 μ_max。

    Parameters
    ----------
    b : ndarray
        观测图像（模糊 + 噪声）。
    kernel : ndarray
        模糊核（用于频域计算）。
    A : BlurOperator
        模糊算子实例。
    lam : float
        LN 正则化参数 α。
    p : float
        Lp 指数（0 < p < 1）。
    mu0, mu_max : float
        HQS 惩罚参数初始值与最大值。
    n_dc : int
        每个 μ 下的 DC 迭代次数。
    n_hqs : int
        每个 DC 迭代内的 HQS 迭代次数。
    """
    m, n = b.shape
    x = b.copy()
    grad_op = GradientOperator((m, n))
    dc = DCLogDecomposition()

    # 预计算频域核
    k_fft = np.fft.fft2(kernel, s=(m, n))
    k_conj = np.conj(k_fft)
    k_sq = np.abs(k_fft)**2
    grad_sq = grad_op.grad_sq_fft

    mu = mu0
    obj_history = []
    outer_iter = 0

    while mu <= mu_max:
        if verbose:
            print(f"\n[DC-HQS-AGST] μ={mu:.4f}, outer_iter={outer_iter}")

        for dc_iter in range(n_dc):
            # ---- DC 修正右端项 ----
            b_tilde = dc.corrected_rhs(b, x, A)
            b_tilde_fft = np.fft.fft2(b_tilde)

            for hqs_iter in range(n_hqs):
                # ---- u-update：AGST 求解 LN 子问题 ----
                dh, dv = grad_op.forward(x)
                u_h = agst(dh, dh, lam / mu, p, J=3)
                u_v = agst(dv, dv, lam / mu, p, J=3)

                # ---- x-update：FFT 闭式求解 ----
                # 计算 μ * ∇^T u 的频域
                u_h_fft = np.fft.fft2(u_h)
                u_v_fft = np.fft.fft2(u_v)
                grad_term = (np.conj(grad_op.grad_h_fft) * u_h_fft +
                             np.conj(grad_op.grad_v_fft) * u_v_fft)

                numerator = k_conj * b_tilde_fft + mu * grad_term
                denominator = k_sq + mu * grad_sq
                x_fft = numerator / denominator
                x = np.real(np.fft.ifft2(x_fft))

                # 裁剪到 [0, 1]
                x = np.clip(x, 0.0, 1.0)

                if return_history:
                    obj = objective_log(x, A, b, lam=lam, p=p,
                                        use_ln=True, grad_op=grad_op)
                    obj_history.append(obj)

                if verbose and hqs_iter % 2 == 0:
                    print(f"  DC[{dc_iter}] HQS[{hqs_iter}] obj={obj_history[-1]:.6e}")

        mu *= 2.0
        outer_iter += 1

    return x, obj_history


def dc_hqs_agst_l0(b, kernel, A, lam, p=0.8, beta=0.002,
                   mu0=0.01, mu_max=10.0,
                   n_dc=2, n_hqs=5,
                   verbose=False, return_history=True):
    """
    DC-HQS-AGST-L0：在 LN 基础上叠加 L0 梯度先验的混合模型。
    对应论文中的 fast nonlinear sparse model（LN + L0）。
    """
    m, n = b.shape
    x = b.copy()
    grad_op = GradientOperator((m, n))
    dc = DCLogDecomposition()

    k_fft = np.fft.fft2(kernel, s=(m, n))
    k_conj = np.conj(k_fft)
    k_sq = np.abs(k_fft)**2
    grad_sq = grad_op.grad_sq_fft

    mu = mu0
    obj_history = []
    outer_iter = 0

    while mu <= mu_max:
        if verbose:
            print(f"\n[DC-HQS-AGST-L0] μ={mu:.4f}, outer_iter={outer_iter}")

        for dc_iter in range(n_dc):
            b_tilde = dc.corrected_rhs(b, x, A)
            b_tilde_fft = np.fft.fft2(b_tilde)

            for hqs_iter in range(n_hqs):
                # u-update (LN)
                dh, dv = grad_op.forward(x)
                u_h = agst(dh, dh, lam / mu, p, J=3)
                u_v = agst(dv, dv, lam / mu, p, J=3)

                # g-update (L0)
                g_h, g_v = hard_threshold_grad(dh, dv, beta / mu)

                # x-update (FFT)
                u_h_fft = np.fft.fft2(u_h)
                u_v_fft = np.fft.fft2(u_v)
                g_h_fft = np.fft.fft2(g_h)
                g_v_fft = np.fft.fft2(g_v)

                grad_term = (np.conj(grad_op.grad_h_fft) * (u_h_fft + g_h_fft) +
                             np.conj(grad_op.grad_v_fft) * (u_v_fft + g_v_fft))

                numerator = k_conj * b_tilde_fft + mu * grad_term
                denominator = k_sq + 2.0 * mu * grad_sq
                x_fft = numerator / denominator
                x = np.real(np.fft.ifft2(x_fft))
                x = np.clip(x, 0.0, 1.0)

                if return_history:
                    obj = objective_log(x, A, b, lam=lam, p=p,
                                        use_ln=True, use_l0=True,
                                        beta=beta, grad_op=grad_op)
                    obj_history.append(obj)

                if verbose and hqs_iter % 2 == 0:
                    print(f"  DC[{dc_iter}] HQS[{hqs_iter}] obj={obj_history[-1]:.6e}")

        mu *= 2.0
        outer_iter += 1

    return x, obj_history


# ==================== 连续化 FISTA（Continuation） ====================

def fista_lasso_continuation(x0, A, b, lam_target, L=None,
                              lam0_factor=10.0, decay=0.7, max_iter_per_lam=50,
                              tol=1e-6, verbose=False):
    """
    FISTA with Continuation：从较大的 λ 逐步降至目标 λ。
    用于 DC 子问题的快速求解，或作为独立基准方法。
    """
    if L is None:
        L = A.lipschitz
    x = x0.copy()
    lam = lam_target * lam0_factor
    history = []
    while lam > lam_target * 1.001:
        x, h = fista_lasso(x, A, b, lam, L=L, max_iter=max_iter_per_lam, tol=tol, verbose=False)
        history.extend(h)
        lam = max(lam_target, lam * decay)
        if verbose:
            print(f"[FISTA-Cont] λ={lam:.4e}, PSNR-like obj={history[-1]:.6e}")
    # 最后精确求解目标 λ
    x, h = fista_lasso(x, A, b, lam_target, L=L, max_iter=max_iter_per_lam * 2, tol=tol, verbose=False)
    history.extend(h)
    return x, history


# ==================== 两阶段基准（v2.0 保留） ====================

def two_stage_strategy(x0, A, b, lam, c=1.0, max_iter_stage1=300,
                       max_iter_stage2=500, tol=1e-6, verbose=False):
    """两阶段策略：FISTA-LASSO 初始化 → PG-Log-L1 精修。"""
    if verbose:
        print("=" * 60 + "\nStage 1: FISTA-LASSO\n" + "=" * 60)
    x1, h1 = fista_lasso(x0, A, b, lam, max_iter=max_iter_stage1, tol=tol, verbose=verbose)
    if verbose:
        print("\n" + "=" * 60 + "\nStage 2: PG-Log-L1\n" + "=" * 60)
    x2, h2 = proximal_gradient_log(x1, A, b, lam, max_iter=max_iter_stage2, tol=tol, verbose=verbose)
    return x2, h1 + h2
