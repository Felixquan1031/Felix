"""
可视化模块（v3.0）：生成论文所需的全部图表。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 12
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 150


def plot_image_comparison_grid(images, titles, ncols=4, cmap='gray', save_path=None):
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
        axes[i].set_title(title)
        axes[i].axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_curves(histories, labels, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for hist, label in zip(histories, labels):
        ax.plot(hist, label=label, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective value')
    ax.set_title('Convergence curves')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_penalties(save_path=None):
    u = np.linspace(-5, 5, 500)
    square = 0.5 * u**2
    c = 1.345
    huber = np.where(np.abs(u) <= c, 0.5 * u**2, c * np.abs(u) - 0.5 * c**2)
    logloss = np.log1p(u**2)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(u, square, label=r'Square: $\frac{1}{2}u^2$', linewidth=2)
    ax.plot(u, huber, label=r'Huber ($c=1.345$)', linewidth=2)
    ax.plot(u, logloss, label=r'Log: $\log(1+u^2)$', linewidth=2)
    ax.set_xlabel('Residual $u$')
    ax.set_ylabel('Loss value')
    ax.set_title('Comparison of data fidelity terms')
    ax.legend(loc='upper center')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_residual_histogram(residuals, labels, save_path=None):
    n = len(residuals)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, res, label in zip(axes, residuals, labels):
        ax.hist(res.flatten(), bins=80, color='steelblue', edgecolor='k', alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel('Residual value')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_lam_vs_quality(lams, psnrs, ssims, save_path=None):
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    color1 = 'tab:blue'
    ax1.set_xlabel(r'$\lambda$ (log scale)')
    ax1.set_xscale('log')
    ax1.set_ylabel('PSNR (dB)', color=color1)
    ax1.plot(lams, psnrs, color=color1, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('SSIM', color=color2)
    ax2.plot(lams, ssims, color=color2, marker='s', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Reconstruction quality vs. regularization parameter')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_p_sensitivity(ps, psnrs, ssims, save_path=None):
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    color1 = 'tab:blue'
    ax1.set_xlabel(r'$p$')
    ax1.set_ylabel('PSNR (dB)', color=color1)
    ax1.plot(ps, psnrs, color=color1, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('SSIM', color=color2)
    ax2.plot(ps, ssims, color=color2, marker='s', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(r'Sensitivity analysis of parameter $p$')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_gst_threshold_curves(lam, p_values, save_path=None):
    v = np.linspace(-2, 2, 400)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    from .operators import agst
    for p in p_values:
        x = agst(v, v, lam, p, J=3)
        ax.plot(v, x, label=f'$p={p}$', linewidth=1.5)
    ax.plot(v, v, 'k--', linewidth=1, alpha=0.5, label='Identity')
    ax.set_xlabel(r'Input $v$')
    ax.set_ylabel(r'AGST output $x$')
    ax.set_title(r'Adaptive Generalized Soft-Thresholding ($\lambda=' + str(lam) + r'$)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dc_decomposition(save_path=None):
    """可视化 DC 分解：log(1+t^2) = t^2 - h(t)。"""
    t = np.linspace(-4, 4, 500)
    log_loss = np.log1p(t**2)
    quadratic = t**2
    h = t**2 - np.log1p(t**2)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(t, log_loss, label=r'$\log(1+t^2)$', linewidth=2)
    ax.plot(t, quadratic, label=r'$t^2$', linewidth=2, linestyle='--')
    ax.plot(t, h, label=r'$h(t)=t^2-\log(1+t^2)$ (convex)', linewidth=2)
    ax.set_xlabel(r'Residual $t$')
    ax.set_ylabel('Function value')
    ax.set_title('DC decomposition of log-loss')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_table_results(results, save_path=None):
    """将定量结果绘制成表格图。"""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis('off')
    col_labels = ['Method', 'PSNR (dB)', 'SSIM', 'Time (s)']
    rows = []
    for name, res in results.items():
        rows.append([name, f"{res['psnr']:.2f}", f"{res['ssim']:.4f}", f"{res['time']:.2f}"])
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='w')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
