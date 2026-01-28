# loss_advanced.py
# Advanced loss system for SPACE with:
# - Learnable uncertainty weighting (Kendall et al.)
# - MS-SSIM, LPIPS, Gradient, Laplacian losses
# - Anchor frame distillation for self-consistency
# - Progressive curriculum scheduling

import math
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_x import CharbonnierLoss, FrequencyLoss, VGGPerceptualLoss


# =============================================================================
# LPIPS Perceptual Loss
# =============================================================================

class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS).

    Better correlates with human perception than VGG features alone.
    Uses pretrained weights from Zhang et al. (2018).

    Requires: pip install lpips
    """

    def __init__(self, net: str = 'alex', spatial: bool = False):
        """
        Args:
            net: Backbone network ('alex', 'vgg', 'squeeze')
            spatial: If True, return per-pixel loss map
        """
        super().__init__()
        try:
            import lpips
        except ImportError:
            raise ImportError("LPIPS requires: pip install lpips")

        self.lpips = lpips.LPIPS(net=net, spatial=spatial)
        self.lpips.eval()

        # Freeze LPIPS weights
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W] in [0, 1]
            target: [B, 3, H, W] in [0, 1]
        Returns:
            Scalar loss (mean over batch)
        """
        # LPIPS expects [-1, 1] range
        pred_scaled = pred * 2 - 1
        target_scaled = target * 2 - 1
        return self.lpips(pred_scaled, target_scaled).mean()


# =============================================================================
# Multi-Scale SSIM Loss
# =============================================================================

class MultiScaleSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity (MS-SSIM).

    Better captures structural information across scales.
    Loss = 1 - MS_SSIM
    """

    def __init__(self,
                 window_size: int = 11,
                 sigma: float = 1.5,
                 weights: List[float] = None,
                 levels: int = 5):
        """
        Args:
            window_size: Gaussian window size
            sigma: Gaussian sigma
            weights: Per-level weights (default from Wang et al. 2003)
            levels: Number of scales
        """
        super().__init__()

        self.window_size = window_size
        self.levels = levels

        # Default MS-SSIM weights from the paper
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]
        self.register_buffer('weights', torch.tensor(weights))

        # Create Gaussian window
        gauss = torch.exp(-torch.arange(window_size).float()
                          .sub(window_size // 2).pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window.expand(3, 1, -1, -1).contiguous())

    def _ssim_components(self,
                         pred: torch.Tensor,
                         target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute luminance/contrast (cs) and full SSIM."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        pad = self.window_size // 2

        mu1 = F.conv2d(pred, self.window, padding=pad, groups=3)
        mu2 = F.conv2d(target, self.window, padding=pad, groups=3)

        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred ** 2, self.window, padding=pad, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, self.window, padding=pad, groups=3) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=pad, groups=3) - mu1_mu2

        # Clamp for numerical stability
        sigma1_sq = sigma1_sq.clamp(min=0)
        sigma2_sq = sigma2_sq.clamp(min=0)

        # Contrast-structure component
        cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

        # Full SSIM
        ssim = ((2 * mu1_mu2 + C1) * cs) / (mu1_sq + mu2_sq + C1)

        return cs.mean(dim=[1, 2, 3]), ssim.mean(dim=[1, 2, 3])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W]
            target: [B, 3, H, W]
        Returns:
            1 - MS_SSIM
        """
        mcs_list = []
        for i in range(self.levels):
            cs, ssim = self._ssim_components(pred, target)

            if i < self.levels - 1:
                mcs_list.append(cs)
                # Downsample for next level
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
            else:
                # Last level: use full SSIM
                mcs_list.append(ssim)

        # Stack and compute weighted product
        mcs = torch.stack(mcs_list, dim=1)  # [B, levels]

        # Clamp to avoid numerical issues with pow
        mcs = mcs.clamp(min=1e-10)

        # MS-SSIM = product of (cs_i ^ weight_i)
        ms_ssim = torch.prod(mcs ** self.weights, dim=1).mean()

        return 1 - ms_ssim


# =============================================================================
# Gradient (Edge) Loss
# =============================================================================

class GradientLoss(nn.Module):
    """
    Gradient loss for edge sharpness.

    Compares Sobel gradients between prediction and target.
    """

    def __init__(self, edge_weight: float = 1.0):
        super().__init__()
        self.edge_weight = edge_weight

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)

        # Expand for 3 channels
        self.register_buffer('sobel_x', sobel_x.expand(3, 1, 3, 3).contiguous())
        self.register_buffer('sobel_y', sobel_y.expand(3, 1, 3, 3).contiguous())

    def _compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Sobel gradients."""
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        return grad_x, grad_y

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W]
            target: [B, 3, H, W]
        Returns:
            Gradient L1 loss
        """
        pred_gx, pred_gy = self._compute_gradients(pred)
        target_gx, target_gy = self._compute_gradients(target)

        loss_x = F.l1_loss(pred_gx, target_gx)
        loss_y = F.l1_loss(pred_gy, target_gy)

        return self.edge_weight * (loss_x + loss_y)


# =============================================================================
# Laplacian Pyramid Loss
# =============================================================================

class LaplacianPyramidLoss(nn.Module):
    """
    Laplacian Pyramid Loss for multi-scale reconstruction.

    Matches across multiple frequency bands.
    """

    def __init__(self,
                 num_levels: int = 5,
                 weights: List[float] = None):
        super().__init__()

        self.num_levels = num_levels

        # Increasing weight for finer details
        if weights is None:
            weights = [1.0 * (2 ** i) for i in range(num_levels)]
        self.weights = weights

        # Gaussian kernel for downsampling
        kernel = torch.tensor([[1, 4, 6, 4, 1],
                               [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6],
                               [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]], dtype=torch.float32) / 256.0
        kernel = kernel.view(1, 1, 5, 5)
        self.register_buffer('kernel', kernel.expand(3, 1, 5, 5).contiguous())

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian downsample."""
        x = F.conv2d(x, self.kernel, padding=2, groups=3)
        return x[:, :, ::2, ::2]

    def _upsample(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Bilinear upsample."""
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def _laplacian_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Build Laplacian pyramid."""
        pyramid = []
        current = x

        for i in range(self.num_levels - 1):
            down = self._downsample(current)
            up = self._upsample(down, current.shape[2:])
            laplacian = current - up
            pyramid.append(laplacian)
            current = down

        # Residual (lowest frequency)
        pyramid.append(current)

        return pyramid

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W]
            target: [B, 3, H, W]
        Returns:
            Weighted sum of L1 losses at each pyramid level
        """
        pred_pyr = self._laplacian_pyramid(pred)
        target_pyr = self._laplacian_pyramid(target)

        loss = 0
        for p, t, w in zip(pred_pyr, target_pyr, self.weights):
            loss = loss + w * F.l1_loss(p, t)

        return loss


# =============================================================================
# Uncertainty Weighting (Kendall et al.)
# =============================================================================

class UncertaintyWeighting(nn.Module):
    """
    Homoscedastic uncertainty weighting for multi-task/multi-loss learning.

    Each loss has a learnable log-variance parameter log(sigma^2).
    Total loss = sum_i (1/(2*sigma_i^2) * L_i + log(sigma_i))

    This automatically balances losses based on task uncertainty.
    Lower uncertainty (sigma) = higher weight for that loss.

    Reference: Kendall et al. "Multi-Task Learning Using Uncertainty to
    Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)
    """

    def __init__(self,
                 loss_names: List[str],
                 init_log_vars: Dict[str, float] = None):
        """
        Args:
            loss_names: Names of losses to weight
            init_log_vars: Initial log(sigma^2) values (default: 0.0 = sigma=1)
        """
        super().__init__()

        self.loss_names = loss_names

        # Initialize log(sigma^2) parameters
        # log(sigma^2) = 0 -> sigma = 1 -> weight = 0.5
        if init_log_vars is None:
            init_log_vars = {name: 0.0 for name in loss_names}

        # Learnable log-variance parameters
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(init_log_vars.get(name, 0.0)))
            for name in loss_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply uncertainty weighting to losses.

        Args:
            losses: Dict mapping loss_name -> loss_value

        Returns:
            total_loss: Weighted sum of losses
            weighted_losses: Dict with individual weighted losses for logging
            weights: Dict with effective weights (precision) for logging
        """
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        weighted = {}
        weights = {}

        for name in self.loss_names:
            if name not in losses:
                continue

            loss = losses[name]
            log_var = self.log_vars[name]

            # precision = 1 / sigma^2 = exp(-log_var)
            precision = torch.exp(-log_var)

            # weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            # The 0.5 * log_var acts as regularization (prevents sigma -> inf)
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var

            total = total + weighted_loss
            weighted[name] = weighted_loss
            weights[name] = precision

        return total, weighted, weights

    def get_weights(self) -> Dict[str, float]:
        """Get current effective weights (precision = 1/sigma^2)."""
        return {
            name: torch.exp(-self.log_vars[name]).item()
            for name in self.loss_names
        }

    def get_sigmas(self) -> Dict[str, float]:
        """Get current sigma values."""
        return {
            name: torch.exp(0.5 * self.log_vars[name]).item()
            for name in self.loss_names
        }


# =============================================================================
# Anchor Frame Distillation Loss
# =============================================================================

class CoordinateAnchorDistillationLoss(nn.Module):
    """
    Anchor distillation that works with coordinate sampling format.

    At anchor timestamps, the model should perfectly reconstruct the input frames.
    This enforces self-consistency: the model must preserve frames it was given.

    Samples coordinates at anchor times and compares with values from input frames.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 num_samples: int = 512,
                 anchor_indices: Union[str, List[int]] = 'all'):
        """
        Args:
            eps: Charbonnier epsilon
            num_samples: Number of coordinates to sample per anchor
            anchor_indices: 'all' for all anchors, or list like [0, -1] for first/last
        """
        super().__init__()
        self.eps = eps
        self.num_samples = num_samples
        self.anchor_indices = anchor_indices

    def forward(self,
                model: nn.Module,
                input_frames: torch.Tensor,
                input_times: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute anchor distillation using coordinate sampling.

        Args:
            model: SPACE model (with forward method)
            input_frames: [B, N, 3, H, W]
            input_times: [B, N]

        Returns:
            Dict with 'anchor_distill' total and per-anchor losses
        """
        B, N, C, H, W = input_frames.shape
        device = input_frames.device

        # Determine which anchors to use
        if self.anchor_indices == 'all':
            indices = list(range(N))
        else:
            indices = [i if i >= 0 else N + i for i in self.anchor_indices]

        losses = {}
        total = torch.tensor(0.0, device=device)

        for idx in indices:
            # Get anchor frame and time
            anchor_frame = input_frames[:, idx]  # [B, 3, H, W]
            anchor_time = input_times[:, idx]    # [B]

            # Sample random coordinates
            x = torch.rand(B, self.num_samples, device=device) * 2 - 1
            y = torch.rand(B, self.num_samples, device=device) * 2 - 1
            t = anchor_time.unsqueeze(1).expand(-1, self.num_samples)

            coords = torch.stack([x, y, t], dim=-1)  # [B, Q, 3]

            # Sample target RGB from anchor frame
            grid = torch.stack([x, y], dim=-1).view(B, self.num_samples, 1, 2)
            target_rgb = F.grid_sample(
                anchor_frame, grid,
                mode='bilinear', padding_mode='border', align_corners=True
            ).squeeze(-1).permute(0, 2, 1)  # [B, Q, 3]

            # Get model prediction
            pred_rgb = model(input_frames, input_times, coords)  # [B, Q, 3]

            # Charbonnier loss
            diff = pred_rgb - target_rgb
            loss = torch.sqrt(diff ** 2 + self.eps ** 2).mean()

            losses[f'anchor_{idx}'] = loss
            total = total + loss

        losses['anchor_distill'] = total / len(indices)
        return losses


# =============================================================================
# Progressive Loss Scheduler
# =============================================================================

class ProgressiveLossScheduler:
    """
    Progressive curriculum for loss activation.

    Start with simple losses, gradually add complex ones.
    """

    def __init__(self,
                 schedule: Dict[int, List[str]] = None):
        """
        Args:
            schedule: Dict mapping epoch -> list of losses to activate
                      e.g., {0: ['charbonnier'], 10: ['charbonnier', 'msssim']}
        """
        if schedule is None:
            schedule = {
                0: ['charbonnier', 'anchor_distill'],
                5: ['charbonnier', 'anchor_distill', 'msssim'],
                15: ['charbonnier', 'anchor_distill', 'msssim', 'gradient'],
                30: ['charbonnier', 'anchor_distill', 'msssim', 'gradient', 'lpips'],
                50: ['charbonnier', 'anchor_distill', 'msssim', 'gradient', 'lpips', 'frequency'],
            }
        self.schedule = schedule

        # Sort epochs
        self.epochs = sorted(schedule.keys())

    def get_active_losses(self, epoch: int) -> List[str]:
        """Get list of active losses for given epoch."""
        active_losses = []
        for e in self.epochs:
            if e <= epoch:
                active_losses = self.schedule[e]
        return active_losses


# =============================================================================
# Advanced SPACE Loss
# =============================================================================

class AdvancedSPACELoss(nn.Module):
    """
    Advanced loss system for SPACE with:
    - Multiple loss components (Charbonnier, MS-SSIM, LPIPS, Gradient, Laplacian, Frequency)
    - Homoscedastic uncertainty weighting
    - Anchor frame distillation
    - Support for both coordinate and full-frame formats
    - Progressive curriculum scheduling
    """

    def __init__(self,
                 # Loss selection
                 use_charbonnier: bool = True,
                 use_msssim: bool = True,
                 use_lpips: bool = True,
                 use_gradient: bool = True,
                 use_laplacian: bool = False,
                 use_frequency: bool = True,
                 use_vgg: bool = False,

                 # Anchor distillation
                 use_anchor_distillation: bool = True,
                 anchor_distill_weight: float = 0.5,
                 anchor_distill_indices: Union[str, List[int]] = 'all',
                 anchor_distill_samples: int = 512,

                 # Uncertainty weighting
                 use_uncertainty_weighting: bool = True,

                 # Fixed weights (used if uncertainty_weighting is False)
                 fixed_weights: Dict[str, float] = None,

                 # Progressive curriculum
                 use_progressive: bool = True,
                 progressive_schedule: Dict[int, List[str]] = None,

                 # Specific loss configs
                 charbonnier_eps: float = 1e-6,
                 lpips_net: str = 'alex',
                 msssim_levels: int = 5,
                 laplacian_levels: int = 5):
        """
        Args:
            use_*: Enable/disable specific losses
            use_anchor_distillation: Enable anchor frame distillation
            anchor_distill_weight: Fixed weight for anchor distillation
            anchor_distill_indices: 'all' or list like [0, -1]
            use_uncertainty_weighting: Use learnable uncertainty weights
            fixed_weights: Dict of fixed weights if not using uncertainty
            use_progressive: Use progressive curriculum
            progressive_schedule: Custom schedule for progressive training
        """
        super().__init__()

        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_anchor_distillation = use_anchor_distillation
        self.anchor_distill_weight = anchor_distill_weight
        self.use_progressive = use_progressive

        # Track which losses are enabled
        self.enabled_losses = {}

        # Initialize loss components
        loss_names = []

        if use_charbonnier:
            self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
            loss_names.append('charbonnier')
            self.enabled_losses['charbonnier'] = True

        if use_msssim:
            self.msssim = MultiScaleSSIMLoss(levels=msssim_levels)
            loss_names.append('msssim')
            self.enabled_losses['msssim'] = True

        if use_lpips:
            self.lpips = LPIPSLoss(net=lpips_net)
            loss_names.append('lpips')
            self.enabled_losses['lpips'] = True

        if use_gradient:
            self.gradient = GradientLoss()
            loss_names.append('gradient')
            self.enabled_losses['gradient'] = True

        if use_laplacian:
            self.laplacian = LaplacianPyramidLoss(num_levels=laplacian_levels)
            loss_names.append('laplacian')
            self.enabled_losses['laplacian'] = True

        if use_frequency:
            self.frequency = FrequencyLoss()
            loss_names.append('frequency')
            self.enabled_losses['frequency'] = True

        if use_vgg:
            self.vgg = VGGPerceptualLoss()
            loss_names.append('vgg')
            self.enabled_losses['vgg'] = True

        self.loss_names = loss_names
        self.active_losses = set(loss_names)  # All active by default

        # Uncertainty weighting
        if use_uncertainty_weighting:
            # Initialize with reasonable priors
            init_log_vars = {
                'charbonnier': 0.0,   # sigma=1, high weight
                'msssim': 0.5,        # sigma~1.3, medium-high
                'lpips': 1.0,         # sigma~1.6, medium
                'gradient': 1.5,      # sigma~2.1, lower
                'laplacian': 1.5,
                'frequency': 2.0,     # sigma~2.7, lowest
                'vgg': 1.0,
            }
            self.uncertainty = UncertaintyWeighting(loss_names, init_log_vars)
        else:
            # Fixed weights
            if fixed_weights is None:
                fixed_weights = {
                    'charbonnier': 1.0,
                    'msssim': 0.5,
                    'lpips': 0.1,
                    'gradient': 0.1,
                    'laplacian': 0.05,
                    'frequency': 0.1,
                    'vgg': 0.1,
                }
            self.fixed_weights = fixed_weights

        # Anchor distillation
        if use_anchor_distillation:
            self.anchor_distill = CoordinateAnchorDistillationLoss(
                eps=charbonnier_eps,
                num_samples=anchor_distill_samples,
                anchor_indices=anchor_distill_indices,
            )

        # Progressive scheduler
        if use_progressive:
            self.progressive_scheduler = ProgressiveLossScheduler(progressive_schedule)

    def set_active_losses(self, active: List[str]):
        """Set which losses are currently active (for progressive curriculum)."""
        self.active_losses = set(active)

    def update_for_epoch(self, epoch: int):
        """Update active losses based on epoch (progressive curriculum)."""
        if self.use_progressive:
            active = self.progressive_scheduler.get_active_losses(epoch)
            self.set_active_losses(active)
            return active
        return list(self.active_losses)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                model: nn.Module = None,
                input_frames: torch.Tensor = None,
                input_times: torch.Tensor = None,
                compute_all: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute advanced loss.

        Args:
            pred: [B, 3, H, W] or [B, Q, 3] predictions
            target: [B, 3, H, W] or [B, Q, 3] targets
            model: SPACE model (required for anchor distillation)
            input_frames: [B, N, 3, H, W] (required for anchor distillation)
            input_times: [B, N] (required for anchor distillation)
            compute_all: If False, only compute coordinate-compatible losses

        Returns:
            Dict with 'total' and individual losses
        """
        losses = {}
        is_coords = pred.dim() == 3 and pred.shape[-1] == 3

        # Coordinate sampling mode: only Charbonnier + Anchor Distillation
        if is_coords:
            if hasattr(self, 'charbonnier') and 'charbonnier' in self.active_losses:
                losses['charbonnier'] = self.charbonnier(pred, target)

            # Anchor distillation with coordinate mode
            if (self.use_anchor_distillation and
                'anchor_distill' in self.active_losses and
                model is not None and
                input_frames is not None):

                anchor_losses = self.anchor_distill(model, input_frames, input_times)
                losses['anchor_distill'] = anchor_losses['anchor_distill']
                # Store individual anchor losses for logging
                for k, v in anchor_losses.items():
                    if k != 'anchor_distill':
                        losses[k] = v

            # Compute total for coordinate mode
            total = torch.tensor(0.0, device=pred.device)
            if 'charbonnier' in losses:
                if self.use_uncertainty_weighting and hasattr(self, 'uncertainty'):
                    weighted_total, _, _ = self.uncertainty({'charbonnier': losses['charbonnier']})
                    total = total + weighted_total
                else:
                    total = total + self.fixed_weights.get('charbonnier', 1.0) * losses['charbonnier']

            # Add anchor distillation (not uncertainty weighted)
            if 'anchor_distill' in losses:
                total = total + self.anchor_distill_weight * losses['anchor_distill']

            losses['total'] = total
            return losses

        # Full-frame mode: compute all active losses
        raw_losses = {}

        if compute_all:
            if hasattr(self, 'charbonnier') and 'charbonnier' in self.active_losses:
                raw_losses['charbonnier'] = self.charbonnier(pred, target)

            if hasattr(self, 'msssim') and 'msssim' in self.active_losses:
                raw_losses['msssim'] = self.msssim(pred, target)

            if hasattr(self, 'lpips') and 'lpips' in self.active_losses:
                raw_losses['lpips'] = self.lpips(pred, target)

            if hasattr(self, 'gradient') and 'gradient' in self.active_losses:
                raw_losses['gradient'] = self.gradient(pred, target)

            if hasattr(self, 'laplacian') and 'laplacian' in self.active_losses:
                raw_losses['laplacian'] = self.laplacian(pred, target)

            if hasattr(self, 'frequency') and 'frequency' in self.active_losses:
                raw_losses['frequency'] = self.frequency(pred, target)

            if hasattr(self, 'vgg') and 'vgg' in self.active_losses:
                raw_losses['vgg'] = self.vgg(pred, target)
        else:
            # Fast mode: only charbonnier
            if hasattr(self, 'charbonnier') and 'charbonnier' in self.active_losses:
                raw_losses['charbonnier'] = self.charbonnier(pred, target)

        # Store raw losses
        losses.update(raw_losses)

        # Apply weighting
        if self.use_uncertainty_weighting and hasattr(self, 'uncertainty'):
            total, weighted_losses, weights = self.uncertainty(raw_losses)
            for name, val in weighted_losses.items():
                losses[f'{name}_weighted'] = val
            losses['_weights'] = weights
        else:
            total = torch.tensor(0.0, device=pred.device)
            for name, loss in raw_losses.items():
                if name in self.fixed_weights:
                    total = total + self.fixed_weights[name] * loss

        losses['total'] = total
        return losses

    def get_uncertainty_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current uncertainty parameters for logging."""
        if not self.use_uncertainty_weighting or not hasattr(self, 'uncertainty'):
            return {}

        return {
            'weights': self.uncertainty.get_weights(),
            'sigmas': self.uncertainty.get_sigmas(),
            'log_vars': {name: p.item() for name, p in self.uncertainty.log_vars.items()}
        }
