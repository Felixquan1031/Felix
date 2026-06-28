"""
实验模块（v3.0）：运行对比实验并收集定量指标。
"""

import numpy as np
import time
from .operators import compute_psnr, compute_ssim
from .algorithms import (
    fista_lasso,
    fista_huber,
    proximal_gradient_log,
    hqs_l2_l1,
    hqs_huber_l1,
    dc_hqs_agst,
    dc_hqs_agst_l0,
    two_stage_strategy,
)


def estimate_sigma_mad(b):
    med = np.median(b)
    mad = np.median(np.abs(b - med))
    return mad / 0.6745


def run_comparison(x_true, kernel, A, b, lam, p=0.8, verbose=False):
    """
    运行八种方法的对比实验：
    1. LASSO (FISTA，图像域)
    2. Huber-LASSO (FISTA，图像域)
    3. Log-PG-L1（固定步长 PG，图像域）
    4. Two-Stage（图像域）
    5. HQS-L2-L1（梯度域 LASSO，公平基准）
    6. HQS-Huber-L1（梯度域 Huber-LASSO，公平基准）
    7. DC-HQS-AGST（本文核心改进，梯度域 LN）
    8. DC-HQS-AGST-L0（梯度域 LN + L0）
    """
    x0 = np.zeros_like(x_true)
    sigma = estimate_sigma_mad(b)
    c = 1.345 * sigma

    results = {}

    # 1. LASSO (图像域，保留作为历史参考)
    if verbose:
        print("\n[1/8] Running LASSO (FISTA, pixel domain)...")
    t0 = time.time()
    x_lasso, hist_lasso = fista_lasso(x0, A, b, lam, max_iter=400, tol=1e-6, verbose=verbose)
    t_lasso = time.time() - t0
    x_lasso = np.clip(x_lasso, 0, 1)
    results['LASSO'] = {
        'x': x_lasso,
        'psnr': compute_psnr(x_true, x_lasso),
        'ssim': compute_ssim(x_true, x_lasso),
        'time': t_lasso,
        'history': hist_lasso,
    }

    # 2. Huber-LASSO (图像域，保留作为历史参考)
    if verbose:
        print(f"\n[2/8] Running Huber-LASSO, c={c:.4f} (pixel domain)...")
    t0 = time.time()
    x_huber, hist_huber = fista_huber(x0, A, b, lam, c=c, max_iter=400, tol=1e-6, verbose=verbose)
    t_huber = time.time() - t0
    x_huber = np.clip(x_huber, 0, 1)
    results['Huber'] = {
        'x': x_huber,
        'psnr': compute_psnr(x_true, x_huber),
        'ssim': compute_ssim(x_true, x_huber),
        'time': t_huber,
        'history': hist_huber,
    }

    # 3. Log-PG-L1
    if verbose:
        print("\n[3/8] Running Log-PG-L1...")
    t0 = time.time()
    x_log, hist_log = proximal_gradient_log(x0, A, b, lam, step=0.5 / A.lipschitz,
                                            max_iter=800, tol=1e-6, verbose=verbose)
    t_log = time.time() - t0
    x_log = np.clip(x_log, 0, 1)
    results['Log-L1'] = {
        'x': x_log,
        'psnr': compute_psnr(x_true, x_log),
        'ssim': compute_ssim(x_true, x_log),
        'time': t_log,
        'history': hist_log,
    }

    # 4. Two-Stage
    if verbose:
        print("\n[4/8] Running Two-Stage (LASSO → Log-L1)...")
    t0 = time.time()
    x_2stage, hist_2stage = two_stage_strategy(
        x0, A, b, lam, c=c, max_iter_stage1=250, max_iter_stage2=500, tol=1e-6, verbose=verbose
    )
    t_2stage = time.time() - t0
    x_2stage = np.clip(x_2stage, 0, 1)
    results['Two-Stage'] = {
        'x': x_2stage,
        'psnr': compute_psnr(x_true, x_2stage),
        'ssim': compute_ssim(x_true, x_2stage),
        'time': t_2stage,
        'history': hist_2stage,
    }

    # 5. HQS-L2-L1（梯度域公平基准）
    if verbose:
        print(f"\n[5/8] Running HQS-L2-L1 (gradient domain, λ={lam})...")
    t0 = time.time()
    x_hqs_l2, hist_hqs_l2 = hqs_l2_l1(
        b, kernel, A, lam=lam,
        mu0=0.005, mu_max=5.0, n_hqs=10,
        verbose=verbose, return_history=True
    )
    t_hqs_l2 = time.time() - t0
    x_hqs_l2 = np.clip(x_hqs_l2, 0, 1)
    results['HQS-L2-L1'] = {
        'x': x_hqs_l2,
        'psnr': compute_psnr(x_true, x_hqs_l2),
        'ssim': compute_ssim(x_true, x_hqs_l2),
        'time': t_hqs_l2,
        'history': hist_hqs_l2,
    }

    # 6. HQS-Huber-L1（梯度域公平基准）
    if verbose:
        print(f"\n[6/8] Running HQS-Huber-L1 (gradient domain, c={c:.4f})...")
    t0 = time.time()
    x_hqs_huber, hist_hqs_huber = hqs_huber_l1(
        b, kernel, A, lam=lam, c=c,
        mu0=0.005, mu_max=5.0, n_hqs=10, n_pg=5,
        verbose=verbose, return_history=True
    )
    t_hqs_huber = time.time() - t0
    x_hqs_huber = np.clip(x_hqs_huber, 0, 1)
    results['HQS-Huber-L1'] = {
        'x': x_hqs_huber,
        'psnr': compute_psnr(x_true, x_hqs_huber),
        'ssim': compute_ssim(x_true, x_hqs_huber),
        'time': t_hqs_huber,
        'history': hist_hqs_huber,
    }

    # 7. DC-HQS-AGST（核心改进）
    if verbose:
        print(f"\n[7/8] Running DC-HQS-AGST (p={p})...")
    t0 = time.time()
    x_dchqs, hist_dchqs = dc_hqs_agst(
        b, kernel, A, lam=lam, p=p,
        mu0=0.005, mu_max=5.0,
        n_dc=2, n_hqs=5,
        verbose=verbose, return_history=True
    )
    t_dchqs = time.time() - t0
    x_dchqs = np.clip(x_dchqs, 0, 1)
    results['DC-HQS-AGST'] = {
        'x': x_dchqs,
        'psnr': compute_psnr(x_true, x_dchqs),
        'ssim': compute_ssim(x_true, x_dchqs),
        'time': t_dchqs,
        'history': hist_dchqs,
    }

    # 8. DC-HQS-AGST-L0（混合模型）
    if verbose:
        print(f"\n[8/8] Running DC-HQS-AGST-L0 (p={p})...")
    beta_l0 = 0.0015
    t0 = time.time()
    x_dchqs_l0, hist_dchqs_l0 = dc_hqs_agst_l0(
        b, kernel, A, lam=lam, p=p, beta=beta_l0,
        mu0=0.005, mu_max=5.0,
        n_dc=2, n_hqs=5,
        verbose=verbose, return_history=True
    )
    t_dchqs_l0 = time.time() - t0
    x_dchqs_l0 = np.clip(x_dchqs_l0, 0, 1)
    results['DC-HQS-AGST-L0'] = {
        'x': x_dchqs_l0,
        'psnr': compute_psnr(x_true, x_dchqs_l0),
        'ssim': compute_ssim(x_true, x_dchqs_l0),
        'time': t_dchqs_l0,
        'history': hist_dchqs_l0,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("Summary of results")
        print("=" * 70)
        for name, res in results.items():
            print(f"{name:18s}: PSNR={res['psnr']:6.2f} dB, SSIM={res['ssim']:.4f}, Time={res['time']:5.2f} s")

    return results


def lambda_sensitivity(x_true, kernel, A, b, lams, p=0.8, verbose=False):
    """测试 DC-HQS-AGST 在不同 λ 下的表现。"""
    psnrs = []
    ssims = []
    for lam in lams:
        if verbose:
            print(f"Testing lambda={lam:.2e}...")
        x, _ = dc_hqs_agst(b, kernel, A, lam=lam, p=p,
                           mu0=0.005, mu_max=5.0,
                           n_dc=2, n_hqs=4,
                           verbose=False, return_history=False)
        x = np.clip(x, 0, 1)
        psnrs.append(compute_psnr(x_true, x))
        ssims.append(compute_ssim(x_true, x))
    return psnrs, ssims


def p_sensitivity(x_true, kernel, A, b, lam, ps, verbose=False):
    """测试 DC-HQS-AGST 在不同 p 值下的表现。"""
    psnrs = []
    ssims = []
    for p in ps:
        if verbose:
            print(f"Testing p={p:.2f}...")
        x, _ = dc_hqs_agst(b, kernel, A, lam=lam, p=p,
                           mu0=0.005, mu_max=5.0,
                           n_dc=2, n_hqs=4,
                           verbose=False, return_history=False)
        x = np.clip(x, 0, 1)
        psnrs.append(compute_psnr(x_true, x))
        ssims.append(compute_ssim(x_true, x))
    return psnrs, ssims
