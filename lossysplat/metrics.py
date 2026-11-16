"""Image metrics. Images are (H, W, 3) tensors in [0, 1]."""
import torch
import torch.nn.functional as F


def psnr(a, b, mask=None):
    """PSNR in dB; with `mask` (H, W) only the selected pixels count."""
    err = (a - b) ** 2
    mse = err.mean() if mask is None else (err * mask[..., None]).sum() / (mask.sum().clamp(min=1) * 3)
    return float(-10.0 * torch.log10(mse.clamp(min=1e-10)))


def foreground_region(alpha_a, alpha_b, dilate=2):
    """Pixels covered by either image's foreground, grown by `dilate` pixels."""
    m = ((alpha_a > 0.5) | (alpha_b > 0.5)).float()[None, None]
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    return m[0, 0]


def _gaussian_window(size, sigma, device, dtype):
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a, b, size=11, sigma=1.5):
    """Mean SSIM (Wang et al. 2004) with a Gaussian window, averaged over channels."""
    x = a.permute(2, 0, 1)[:, None]
    y = b.permute(2, 0, 1)[:, None]
    w = _gaussian_window(size, sigma, a.device, a.dtype)
    pad = size // 2
    mu_x, mu_y = F.conv2d(x, w, padding=pad), F.conv2d(y, w, padding=pad)
    var_x = F.conv2d(x * x, w, padding=pad) - mu_x**2
    var_y = F.conv2d(y * y, w, padding=pad) - mu_y**2
    cov = F.conv2d(x * y, w, padding=pad) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(s.mean())


class Lpips:
    """LPIPS (AlexNet) wrapper; the network is loaded on first use."""

    def __init__(self, device):
        import lpips
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, a, b):
        x = a.permute(2, 0, 1)[None] * 2 - 1
        y = b.permute(2, 0, 1)[None] * 2 - 1
        return float(self.net(x, y))
