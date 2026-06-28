#!/usr/bin/env python3
"""
主脚本（v3.0）：运行图像去模糊对比实验，生成论文所需的全部图表。

运行方式:
    python main.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.data import generate_deblurring_data
from src.operators import BlurOperator, compute_psnr
from src.experiments import run_comparison, lambda_sensitivity, p_sensitivity
from src.visualize import (
    plot_image_comparison_grid,
    plot_loss_curves,
    plot_loss_penalties,
    plot_residual_histogram,
    plot_lam_vs_quality,
    plot_p_sensitivity,
    plot_gst_threshold_curves,
    plot_dc_decomposition,
    plot_table_results,
)


def main():
    fig_dir = 'figures'
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("Generating synthetic deblurring data...")
    print("=" * 60)

    x_true, kernel, b = generate_deblurring_data(
        size=(256, 256),
        blur_sigma=1.5,
        blur_size=15,
        noise_amount=0.07,
        seed=42
    )

    A = BlurOperator(kernel, x_true.shape)
    print(f"Image shape: {x_true.shape}")
    print(f"Blur kernel shape: {kernel.shape}")
    print(f"Operator Lipschitz constant L = {A.lipschitz:.4f}")

    lam = 0.015
    p = 0.8

    print("\n" + "=" * 60)
    print(f"Running comparison experiments (lambda={lam}, p={p})...")
    print("=" * 60)

    results = run_comparison(x_true, kernel, A, b, lam, p=p, verbose=True)

    # ---- 图 1：重建结果对比 ----
    images = [
        x_true,
        b,
        results['HQS-L2-L1']['x'],
        results['HQS-Huber-L1']['x'],
        results['Log-L1']['x'],
        results['DC-HQS-AGST']['x'],
        results['DC-HQS-AGST-L0']['x'],
    ]
    titles = [
        'Ground Truth',
        f'Blurred & Noisy\n(PSNR: {compute_psnr(x_true, b):.1f} dB)',
        f'HQS-L2-L1\n({results["HQS-L2-L1"]["psnr"]:.1f} dB / {results["HQS-L2-L1"]["ssim"]:.3f})',
        f'HQS-Huber-L1\n({results["HQS-Huber-L1"]["psnr"]:.1f} dB / {results["HQS-Huber-L1"]["ssim"]:.3f})',
        f'Log-PG-L1\n({results["Log-L1"]["psnr"]:.1f} dB / {results["Log-L1"]["ssim"]:.3f})',
        f'DC-HQS-AGST (Ours)\n({results["DC-HQS-AGST"]["psnr"]:.1f} dB / {results["DC-HQS-AGST"]["ssim"]:.3f})',
        f'DC-HQS-AGST-L0 (Ours+)\n({results["DC-HQS-AGST-L0"]["psnr"]:.1f} dB / {results["DC-HQS-AGST-L0"]["ssim"]:.3f})',
    ]
    plot_image_comparison_grid(images, titles, ncols=4,
                                save_path=os.path.join(fig_dir, '01_reconstruction_comparison.png'))
    print(f"[Saved] {fig_dir}/01_reconstruction_comparison.png")

    # ---- 图 2：损失函数惩罚曲线 ----
    plot_loss_penalties(save_path=os.path.join(fig_dir, '02_loss_penalties.png'))
    print(f"[Saved] {fig_dir}/02_loss_penalties.png")

    # ---- 图 3：DC 分解可视化 ----
    plot_dc_decomposition(save_path=os.path.join(fig_dir, '03_dc_decomposition.png'))
    print(f"[Saved] {fig_dir}/03_dc_decomposition.png")

    # ---- 图 4：GST / AGST 阈值曲线 ----
    plot_gst_threshold_curves(lam=0.05, p_values=[0.3, 0.5, 0.7, 1.0],
                               save_path=os.path.join(fig_dir, '04_agst_thresholds.png'))
    print(f"[Saved] {fig_dir}/04_agst_thresholds.png")

    # ---- 图 5：收敛曲线 ----
    histories = [
        results['HQS-L2-L1']['history'],
        results['HQS-Huber-L1']['history'],
        results['Log-L1']['history'],
        results['DC-HQS-AGST']['history'],
    ]
    labels = ['HQS-L2-L1', 'HQS-Huber-L1', 'Log-PG-L1', 'DC-HQS-AGST (Ours)']
    plot_loss_curves(histories, labels, save_path=os.path.join(fig_dir, '05_convergence.png'))
    print(f"[Saved] {fig_dir}/05_convergence.png")

    # ---- 图 6：残差直方图 ----
    residuals = [
        A.residual(results['HQS-L2-L1']['x'], b),
        A.residual(results['HQS-Huber-L1']['x'], b),
        A.residual(results['DC-HQS-AGST']['x'], b),
    ]
    res_labels = ['HQS-L2-L1', 'HQS-Huber-L1', 'DC-HQS-AGST (Ours)']
    plot_residual_histogram(residuals, res_labels,
                             save_path=os.path.join(fig_dir, '06_residual_histograms.png'))
    print(f"[Saved] {fig_dir}/06_residual_histograms.png")

    # ---- 图 7：定量结果表格 ----
    plot_table_results(results, save_path=os.path.join(fig_dir, '07_results_table.png'))
    print(f"[Saved] {fig_dir}/07_results_table.png")

    # ---- 图 8：lambda 敏感性分析 ----
    print("\n" + "=" * 60)
    print("Running lambda sensitivity analysis...")
    print("=" * 60)
    lams = np.logspace(-3, -1, 10)
    psnrs_lam, ssims_lam = lambda_sensitivity(x_true, kernel, A, b, lams, p=p, verbose=True)
    plot_lam_vs_quality(lams, psnrs_lam, ssims_lam,
                         save_path=os.path.join(fig_dir, '08_lambda_sensitivity.png'))
    print(f"[Saved] {fig_dir}/08_lambda_sensitivity.png")

    # ---- 图 9：p 值敏感性分析 ----
    print("\n" + "=" * 60)
    print("Running p sensitivity analysis...")
    print("=" * 60)
    ps = np.linspace(0.3, 1.0, 8)
    psnrs_p, ssims_p = p_sensitivity(x_true, kernel, A, b, lam, ps, verbose=True)
    plot_p_sensitivity(ps, psnrs_p, ssims_p,
                        save_path=os.path.join(fig_dir, '09_p_sensitivity.png'))
    print(f"[Saved] {fig_dir}/09_p_sensitivity.png")

    print("\n" + "=" * 60)
    print("All experiments completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
