# loss_x.py
# Advanced loss functions for SPACE-X training

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.

    Compares high-level features rather than raw pixels,
    leading to more perceptually pleasing results.
    """

    def __init__(self,
                 layers: list = None,
                 weights: list = None,
                 normalize: bool = True):
        super().__init__()

        if layers is None:
            layers = [3, 8, 17, 26, 35]  # relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.layers = layers
        self.weights = weights
        self.normalize = normalize

        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()

        # Freeze
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x

    def _extract_features(self, x: torch.Tensor) -> list:
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W] predicted image
            target: [B, 3, H, W] ground truth image

        Returns:
            Perceptual loss value
        """
        pred = self._normalize(pred)
        target = self._normalize(target)

        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        loss = 0
        for pf, tf, w in zip(pred_features, target_features, self.weights):
            loss = loss + w * F.l1_loss(pf, tf)

        return loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1).

    More robust to outliers than L1, differentiable everywhere.
    L(x) = sqrt(x^2 + eps^2)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff ** 2 + self.eps ** 2).mean()


class FrequencyLoss(nn.Module):
    """
    Frequency-domain loss.

    Penalizes differences in Fourier spectrum, helping with
    high-frequency detail reconstruction.
    """

    def __init__(self, weight_high_freq: float = 2.0):
        super().__init__()
        self.weight_high_freq = weight_high_freq

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W]
            target: [B, 3, H, W]
        """
        # FFT
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        # Magnitude
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Create high-frequency weighting mask
        B, C, H, W = pred_mag.shape
        freq_weight = self._create_freq_weight(H, W, pred.device)

        # Weighted L1 loss on magnitudes
        diff = (pred_mag - target_mag).abs()
        weighted_diff = diff * freq_weight

        return weighted_diff.mean()

    def _create_freq_weight(self, H: int, W: int, device) -> torch.Tensor:
        """Create weight mask that emphasizes high frequencies."""
        # Distance from DC component (center in shifted FFT, corner in unshifted)
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Normalize to [0, 1]
        dist = torch.sqrt((yy / H) ** 2 + (xx / W) ** 2)
        dist = dist / dist.max()

        # Weight: higher for high frequencies
        weight = 1.0 + (self.weight_high_freq - 1.0) * dist

        return weight.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) loss.

    Measures structural similarity between images.
    Loss = 1 - SSIM (so minimizing maximizes SSIM)
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size

        # Create Gaussian window
        gauss = torch.exp(-torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window.expand(3, 1, -1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        pad = self.window_size // 2

        # Ensure window is on same device as input
        window = self.window.to(pred.device)

        mu1 = F.conv2d(pred, window, padding=pad, groups=3)
        mu2 = F.conv2d(target, window, padding=pad, groups=3)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred ** 2, window, padding=pad, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=pad, groups=3) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=3) - mu1_mu2

        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim.mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss.

    Encourages smooth changes between adjacent predicted frames.
    Uses warping with optical flow if available.
    """

    def __init__(self, use_flow: bool = False):
        super().__init__()
        self.use_flow = use_flow

    def forward(self,
                pred_t0: torch.Tensor,
                pred_t1: torch.Tensor,
                flow: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_t0: [B, 3, H, W] prediction at time t
            pred_t1: [B, 3, H, W] prediction at time t+dt
            flow: [B, 2, H, W] optical flow from t to t+1 (optional)
        """
        if self.use_flow and flow is not None:
            # Warp pred_t0 to pred_t1 using flow
            warped = self._warp(pred_t0, flow)
            return F.l1_loss(warped, pred_t1)
        else:
            # Simple temporal gradient penalty
            diff = (pred_t1 - pred_t0).abs()
            return diff.mean()

    def _warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp image x using optical flow."""
        B, C, H, W = x.shape

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Add flow (normalize to [-1, 1])
        flow_normalized = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        flow_normalized[:, :, :, 0] = flow_normalized[:, :, :, 0] / (W / 2)
        flow_normalized[:, :, :, 1] = flow_normalized[:, :, :, 1] / (H / 2)

        sample_grid = grid + flow_normalized

        return F.grid_sample(x, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)


class SPACELoss(nn.Module):
    """
    Combined loss function for SPACE training.

    Combines:
    - Charbonnier (robust reconstruction)
    - SSIM (structural)
    - Perceptual (high-level features)
    - Frequency (high-frequency details)
    """

    def __init__(self,
                 charbonnier_weight: float = 1.0,
                 ssim_weight: float = 0.5,
                 perceptual_weight: float = 0.1,
                 frequency_weight: float = 0.1,
                 use_perceptual: bool = True):
        super().__init__()

        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.frequency_weight = frequency_weight

        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.frequency = FrequencyLoss()

        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = VGGPerceptualLoss()

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                compute_all: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: [B, 3, H, W] or [B, Q, 3] predicted
            target: [B, 3, H, W] or [B, Q, 3] ground truth
            compute_all: If False, only compute reconstruction loss (faster)

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # Handle coordinate-sampled format [B, Q, 3]
        is_coords = pred.dim() == 3 and pred.shape[-1] == 3

        if is_coords:
            # For coordinate sampling, only Charbonnier makes sense
            losses['charbonnier'] = self.charbonnier(pred, target)
            losses['total'] = self.charbonnier_weight * losses['charbonnier']
            return losses

        # Full image format [B, 3, H, W]
        losses['charbonnier'] = self.charbonnier(pred, target)

        if compute_all:
            losses['ssim'] = self.ssim(pred, target)
            losses['frequency'] = self.frequency(pred, target)

            if self.use_perceptual:
                losses['perceptual'] = self.perceptual(pred, target)
            else:
                losses['perceptual'] = torch.tensor(0.0, device=pred.device)
        else:
            losses['ssim'] = torch.tensor(0.0, device=pred.device)
            losses['frequency'] = torch.tensor(0.0, device=pred.device)
            losses['perceptual'] = torch.tensor(0.0, device=pred.device)

        # Total
        losses['total'] = (
            self.charbonnier_weight * losses['charbonnier'] +
            self.ssim_weight * losses['ssim'] +
            self.perceptual_weight * losses['perceptual'] +
            self.frequency_weight * losses['frequency']
        )

        return losses
