# encoder_x.py
# Enhanced Frame Encoder with ConvNeXt backbone and multi-scale features

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (B, C, H, W)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with depthwise conv + inverted bottleneck.

    Architecture:
        - 7x7 depthwise conv
        - LayerNorm
        - 1x1 conv (expand 4x)
        - GELU
        - 1x1 conv (project back)
        - Residual connection with layer scale
    """

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)

        # Layer scale
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma[:, None, None] * x

        x = shortcut + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt-based encoder with multi-scale feature extraction.

    Returns features at 4 scales (1/4, 1/8, 1/16, 1/32) for rich spatial info.

    Based on "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
    """

    def __init__(self,
                 in_channels: int = 3,
                 dims: List[int] = None,
                 depths: List[int] = None,
                 drop_path_rate: float = 0.1,
                 use_gradient_checkpointing: bool = False):
        """
        Args:
            in_channels: Input image channels
            dims: Channel dimensions for each stage [96, 192, 384, 768]
            depths: Number of blocks per stage [3, 3, 9, 3]
            drop_path_rate: Stochastic depth rate
            use_gradient_checkpointing: Use gradient checkpointing to save memory
        """
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [3, 3, 9, 3]

        self.dims = dims
        self.num_stages = len(dims)

        # Stem: patchify with 4x4 conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(self.num_stages):
            # ConvNeXt blocks
            stage = nn.Sequential(*[
                ConvNeXtBlock(dims[i], drop_path=dp_rates[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

            # Downsample (except last stage)
            if i < self.num_stages - 1:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsamples.append(downsample)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input image

        Returns:
            List of features at each scale:
            - scale 0: [B, dims[0], H/4, W/4]
            - scale 1: [B, dims[1], H/8, W/8]
            - scale 2: [B, dims[2], H/16, W/16]
            - scale 3: [B, dims[3], H/32, W/32]
        """
        features = []

        x = self.stem(x)

        for i, stage in enumerate(self.stages):
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            features.append(x)

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        return features

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing."""
        self.use_gradient_checkpointing = enable

    def forward_single_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the final scale feature (for backward compatibility)."""
        features = self.forward(x)
        return features[-1]


class MultiScaleFrameEncoder(nn.Module):
    """
    Multi-scale frame encoder using ConvNeXt backbone.

    Encodes N input frames and returns multi-scale feature pyramids.
    """

    def __init__(self,
                 dims: List[int] = None,
                 depths: List[int] = None,
                 drop_path_rate: float = 0.1,
                 use_gradient_checkpointing: bool = False):
        super().__init__()

        if dims is None:
            dims = [64, 128, 256, 512]  # Smaller than standard ConvNeXt
        if depths is None:
            depths = [2, 2, 6, 2]

        self.encoder = ConvNeXtEncoder(
            in_channels=3,
            dims=dims,
            depths=depths,
            drop_path_rate=drop_path_rate,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.dims = dims

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing."""
        self.encoder.set_gradient_checkpointing(enable)

    def forward(self, frames: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            frames: [B, N, 3, H, W] - N input frames

        Returns:
            List of 4 tensors, each [B, N, C_i, H_i, W_i] for each scale
        """
        B, N, C, H, W = frames.shape

        # Flatten batch and frame dims
        frames_flat = frames.view(B * N, C, H, W)

        # Get multi-scale features
        features = self.encoder(frames_flat)  # List of [B*N, C_i, H_i, W_i]

        # Reshape back to include frame dimension
        multi_scale = []
        for feat in features:
            _, C_i, H_i, W_i = feat.shape
            feat = feat.view(B, N, C_i, H_i, W_i)
            multi_scale.append(feat)

        return multi_scale

    def get_output_dims(self) -> List[int]:
        """Return channel dimensions for each scale."""
        return self.dims


class FrameEncoderX(nn.Module):
    """
    Enhanced frame encoder that returns both:
    1. Multi-scale feature grids (for spatial detail)
    2. Global latent vector (for scene understanding)
    """

    def __init__(self,
                 latent_dim: int = 256,
                 dims: List[int] = None,
                 depths: List[int] = None,
                 use_gradient_checkpointing: bool = False):
        super().__init__()

        if dims is None:
            dims = [64, 128, 256, 512]
        if depths is None:
            depths = [2, 2, 6, 2]

        self.multi_scale_encoder = MultiScaleFrameEncoder(
            dims, depths, use_gradient_checkpointing=use_gradient_checkpointing
        )
        self.dims = dims

        # Global pooling + projection for latent vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.latent_proj = nn.Sequential(
            nn.Linear(dims[-1], latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing."""
        self.multi_scale_encoder.set_gradient_checkpointing(enable)

    def forward(self, frames: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            frames: [B, N, 3, H, W]

        Returns:
            multi_scale: List of [B, N, C_i, H_i, W_i] feature maps
            latent: [B, N, latent_dim] global latent vectors
        """
        B, N = frames.shape[:2]

        # Get multi-scale features
        multi_scale = self.multi_scale_encoder(frames)

        # Get global latent from finest scale
        finest = multi_scale[-1]  # [B, N, C, H, W]
        finest_flat = finest.view(B * N, *finest.shape[2:])
        pooled = self.global_pool(finest_flat).flatten(1)  # [B*N, C]
        latent = self.latent_proj(pooled).view(B, N, -1)  # [B, N, latent_dim]

        return multi_scale, latent
