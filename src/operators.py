"""
算子模块（v3.0）：
- BlurOperator：基于 FFT 的循环卷积模糊算子
- 梯度算子：前向差分（∇）与伴随（∇^T）
- DC 分解：对数损失的凸-凹分解
- AGST：带精确闭式阈值的 Adaptive Generalized Soft-Thresholding
- 硬阈值：L0 梯度先验的闭式解
- 各类损失函数、目标函数与评价指标
"""

import numpy as np
from skimage.metrics import structural_similarity


# ==================== 模糊算子 ====================

class BlurOperator:
    """基于 FFT 的循环卷积模糊算子。"""
    def __init__(self, kernel, image_shape):
        self.image_shape = image_shape
        self.kernel = kernel
        self.k_fft = np.fft.fft2(kernel, s=image_shape)
        self.k_fft_conj = np.conj(self.k_fft)
        self.lipschitz = np.max(np.abs(self.k_fft)**2)

    def forward(self, x):
        x_fft = np.fft.fft2(x)
        return np.real(np.fft.ifft2(x_fft * self.k_fft))

    def adjoint(self, y):
        y_fft = np.fft.fft2(y)
        return np.real(np.fft.ifft2(y_fft * self.k_fft_conj))

    def residual(self, x, b):
        return self.forward(x) - b

    def matvec(self, x):
        return self.forward(x)

    def rmatvec(self, x):
        return self.adjoint(x)


# ==================== 图像梯度算子 ====================

class GradientOperator:
    """
    二维前向差分梯度算子（循环边界），及其频域表示。
    用于 HQS 框架中的闭式 FFT 求解。
    """
    def __init__(self, image_shape):
        self.image_shape = image_shape
        m, n = image_shape
        # 水平前向差分：x[i,j+1] - x[i,j]
        freq_h = np.fft.fftfreq(n).reshape(1, n)
        self.grad_h_fft = np.exp(2j * np.pi * freq_h) - 1.0
        # 垂直前向差分：x[i+1,j] - x[i,j]
        freq_v = np.fft.fftfreq(m).reshape(m, 1)
        self.grad_v_fft = np.exp(2j * np.pi * freq_v) - 1.0
        # 梯度能量谱 |∇_h|^2 + |∇_v|^2
        self.grad_sq_fft = np.abs(self.grad_h_fft)**2 + np.abs(self.grad_v_fft)**2
        # 极小值保护，避免除零
        self.grad_sq_fft = np.maximum(self.grad_sq_fft, 1e-12)

    def forward(self, x):
        """前向差分梯度 (∇_h x, ∇_v x)。"""
        dh = np.roll(x, -1, axis=1) - x
        dv = np.roll(x, -1, axis=0) - x
        return dh, dv

    def adjoint(self, dh, dv):
        """梯度伴随 ∇^T (dh, dv) = ∇_h^T dh + ∇_v^T dv（后向差分）。"""
        dht = np.roll(dh, 1, axis=1) - dh
        dvt = np.roll(dv, 1, axis=0) - dv
        return dht + dvt

    def adjoint_single(self, u_h, u_v):
        """兼容单个数组的伴随（等价于 adjoint）。"""
        return self.adjoint(u_h, u_v)


# ==================== DC 分解 ====================

class DCLogDecomposition:
    """
    对数损失的凸-凹分解（Difference of Convex）。
    log(1+t^2) = t^2 - h(t), 其中 h(t) = t^2 - log(1+t^2) 为凸函数。
    """
    @staticmethod
    def h(t):
        """凸函数 h(t) = t^2 - log(1+t^2)。"""
        return t**2 - np.log1p(t**2)

    @staticmethod
    def grad_h(t):
        """h'(t) = 2t - 2t/(1+t^2) = 2t^3/(1+t^2)。"""
        return 2.0 * t**3 / (1.0 + t**2)

    @staticmethod
    def corrected_rhs(b, x, A):
        """
        计算 DC 迭代中的修正右端项：
            b̃ = b + 0.5 * ∇h(r) = b + r^3/(1+r^2),
        其中 r = Ax - b。
        """
        r = A.residual(x, b)
        return b + r**3 / (1.0 + r**2)


# ==================== 近端算子 ====================

def soft_threshold(v, lam):
    """软阈值算子。"""
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def gst(v, lam, p, J=3):
    """
    Generalized Soft-Thresholding (GST)。
    迭代求解 min_x 0.5*(v-x)^2 + lam*|x|^p 的近似解。
    """
    if p >= 0.99:
        return soft_threshold(v, lam)
    abs_v = np.abs(v)
    b = abs_v.copy()
    for _ in range(J):
        b = abs_v - lam * p * np.power(np.maximum(b, 1e-12), p - 1.0)
        b = np.maximum(b, 0.0)
    return np.sign(v) * b


def agst(v, v_pre, lam, p, J=3):
    """
    Adaptive Generalized Soft-Thresholding (AGST) —— 论文精确实现。
    基于 Zhang et al. 2025, J. Imaging, 11, 327 的 Algorithm 1。
    利用 ||v_pre||_inf 自适应调节正则化强度，并加入闭式阈值 τ。
    """
    v_inf = np.max(np.abs(v_pre))
    if v_inf < 1e-12:
        return gst(v, lam, p, J) if p < 1.0 else soft_threshold(v, lam)

    lam_eff = lam / v_inf
    # 上下限保护，防止自适应参数过度偏离
    lam_eff = np.clip(lam_eff, 0.5 * lam, 5.0 * lam)

    if p >= 0.99:
        return soft_threshold(v, lam_eff)

    abs_v = np.abs(v)
    # 论文公式 (10)：非零极小值点 Bi_τ
    Bi_tau = (lam_eff * (1.0 - p)) ** (1.0 / (2.0 - p))
    # 论文公式 (11)：阈值 τ
    tau = Bi_tau + (lam_eff * p / 2.0) * (Bi_tau ** (p - 1.0))

    # 硬阈值：小于 τ 的直接置零
    mask = abs_v > tau
    result = np.zeros_like(v)
    if np.any(mask):
        # 对超过阈值的元素执行 GST
        b = abs_v[mask].copy()
        for _ in range(J):
            b = abs_v[mask] - lam_eff * p * np.power(np.maximum(b, 1e-12), p - 1.0)
            b = np.maximum(b, 0.0)
        result[mask] = b
    return np.sign(v) * result


def hard_threshold_grad(dh, dv, threshold_sq):
    """
    L0 梯度先验的闭式解（论文公式 23）。
    g = ∇x  if |∇x|^2 > threshold_sq
        0   otherwise
    """
    grad_sq = dh**2 + dv**2
    mask = grad_sq > threshold_sq
    gh = np.where(mask, dh, 0.0)
    gv = np.where(mask, dv, 0.0)
    return gh, gv


# ==================== 损失函数与目标函数 ====================

def loss_log(r):
    return np.log1p(r**2)


def grad_loss_log(r):
    return 2.0 * r / (1.0 + r**2)


def objective_log(x, A, b, lam=0.0, p=1.0, use_ln=False, use_l0=False, beta=0.0, grad_op=None):
    """
    计算目标函数值：
        F(x) = sum log(1+(Ax-b)^2) + lam * R(x)
    其中 R(x) 可以是 L1、Lp 或 LN（梯度域）。
    """
    r = A.residual(x, b)
    f_val = np.sum(loss_log(r))

    if lam > 0 and grad_op is not None:
        dh, dv = grad_op.forward(x)
        if use_ln and p < 1.0:
            grad_inf = np.max(np.abs(dh)) + np.max(np.abs(dv))
            if grad_inf > 1e-12:
                f_val += lam * (np.sum(np.abs(dh)**p) + np.sum(np.abs(dv)**p)) / grad_inf
        else:
            # 默认图像域 L1（用于基准方法）
            f_val += lam * np.sum(np.abs(x))

    if use_l0 and beta > 0 and grad_op is not None:
        dh, dv = grad_op.forward(x)
        f_val += beta * np.sum((dh**2 + dv**2) > 1e-12)

    return f_val


# ==================== 评价指标 ====================

def compute_psnr(x_true, x_rec):
    mse = np.mean((x_true - x_rec)**2)
    if mse == 0:
        return np.inf
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(x_true, x_rec):
    return structural_similarity(x_true, x_rec, data_range=1.0)
