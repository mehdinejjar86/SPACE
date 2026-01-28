# decoder_x.py
# Enhanced decoder with multi-scale feature grids, skip connections, and hybrid architecture

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding with learnable frequencies.

    Better than fixed RFF - learns optimal frequency distribution.
    """

    def __init__(self,
                 coord_dim: int = 3,
                 embed_dim: int = 256,
                 num_frequencies: int = 64,
                 learnable: bool = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies

        if learnable:
            # Learnable frequency matrix
            self.frequencies = nn.Parameter(
                torch.randn(coord_dim, num_frequencies) * 10.0
            )
        else:
            # Fixed log-linear frequencies
            freqs = 2.0 ** torch.linspace(0, num_frequencies // coord_dim - 1, num_frequencies // coord_dim)
            freqs = freqs.unsqueeze(0).expand(coord_dim, -1).reshape(coord_dim, -1)
            self.register_buffer('frequencies', freqs)

        # Project to embed_dim
        self.proj = nn.Linear(num_frequencies * 2, embed_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [..., coord_dim]

        Returns:
            [..., embed_dim]
        """
        # Project coordinates to frequencies
        proj = coords @ self.frequencies  # [..., num_frequencies]

        # Sin/cos encoding
        encoded = torch.cat([torch.sin(2 * math.pi * proj),
                             torch.cos(2 * math.pi * proj)], dim=-1)

        return self.proj(encoded)


class FeatureGridSampler(nn.Module):
    """
    Sample features from multi-scale feature grids at query coordinates.

    Combines features from multiple scales with learned weighting.
    """

    def __init__(self, dims: List[int], output_dim: int = 256):
        super().__init__()

        self.dims = dims
        self.num_scales = len(dims)

        # Per-scale projection
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in dims
        ])

        # Scale weighting (learned)
        self.scale_weights = nn.Sequential(
            nn.Linear(3, 64),  # Input: (x, y, t)
            nn.GELU(),
            nn.Linear(64, len(dims)),
            nn.Softmax(dim=-1),
        )

        # Output fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * len(dims), output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self,
                feature_grids: List[torch.Tensor],
                coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_grids: List of [B, C_i, H_i, W_i] feature maps
            coords: [B, Q, 3] query coordinates (x, y, t) in [-1, 1]

        Returns:
            [B, Q, output_dim] sampled and fused features
        """
        B, Q, _ = coords.shape

        # Extract spatial coords for sampling
        xy = coords[:, :, :2]  # [B, Q, 2]
        grid = xy.view(B, Q, 1, 2)  # [B, Q, 1, 2] for grid_sample

        # Sample from each scale
        sampled = []
        for i, (features, proj) in enumerate(zip(feature_grids, self.projections)):
            # Sample: [B, C_i, Q, 1]
            s = F.grid_sample(features, grid, mode='bilinear',
                              padding_mode='border', align_corners=True)
            s = s.squeeze(-1).permute(0, 2, 1)  # [B, Q, C_i]
            s = proj(s)  # [B, Q, output_dim]
            sampled.append(s)

        # Concatenate and fuse
        concat = torch.cat(sampled, dim=-1)  # [B, Q, output_dim * num_scales]
        fused = self.fusion(concat)

        return fused


class ModulatedSIREN(nn.Module):
    """
    SIREN with feature modulation at each layer.

    Instead of just bias modulation, also modulates activations
    based on sampled spatial features.
    """

    def __init__(self,
                 coord_dim: int = 256,
                 hidden_dim: int = 256,
                 feature_dim: int = 256,
                 num_layers: int = 4,
                 omega_0: float = 30.0):
        super().__init__()

        self.omega_0 = omega_0
        self.num_layers = num_layers

        # SIREN layers
        self.layers = nn.ModuleList()
        dims = [coord_dim] + [hidden_dim] * (num_layers - 1) + [3]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        # Feature modulation networks (FiLM-style)
        self.film_generators = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.film_generators.append(nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dims[i + 1] * 2),  # gamma and beta
            ))

        self._init_weights()

    def _init_weights(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                bound = 1.0 / layer.in_features
            else:
                bound = math.sqrt(6.0 / layer.in_features) / self.omega_0

            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)

        # Initialize FiLM to identity
        for gen in self.film_generators:
            nn.init.zeros_(gen[-1].weight)
            # gamma=1, beta=0
            nn.init.ones_(gen[-1].bias[:gen[-1].bias.shape[0] // 2])
            nn.init.zeros_(gen[-1].bias[gen[-1].bias.shape[0] // 2:])

    def forward(self,
                coord_features: torch.Tensor,
                spatial_features: torch.Tensor,
                biases: List[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            coord_features: [B, Q, coord_dim] encoded coordinates
            spatial_features: [B, Q, feature_dim] sampled from feature grids
            biases: Optional list of [B, dim] scene-level biases

        Returns:
            [B, Q, 3] RGB values
        """
        x = coord_features

        for i, (layer, film_gen) in enumerate(zip(self.layers, self.film_generators)):
            is_last = (i == len(self.layers) - 1)

            # Linear transformation
            x = layer(x)

            # Add scene-level bias if provided
            if biases is not None and i < len(biases):
                x = x + biases[i].unsqueeze(1)

            # FiLM modulation from spatial features
            film_params = film_gen(spatial_features)  # [B, Q, dim*2]
            gamma, beta = film_params.chunk(2, dim=-1)
            x = gamma * x + beta

            # Activation
            if not is_last:
                x = torch.sin(self.omega_0 * x)

        return torch.sigmoid(x)


class HybridDecoder(nn.Module):
    """
    Hybrid decoder combining:
    1. Coordinate-based SIREN for fine details
    2. Feature grid sampling for spatial context
    3. Skip connections from encoder features
    """

    def __init__(self,
                 coord_dim: int = 256,
                 hidden_dim: int = 256,
                 feature_dims: List[int] = None,
                 num_siren_layers: int = 4,
                 omega_0: float = 30.0):
        super().__init__()

        if feature_dims is None:
            feature_dims = [64, 128, 256, 512]

        # Coordinate encoder
        self.coord_encoder = FourierFeatures(
            coord_dim=3,
            embed_dim=coord_dim,
            num_frequencies=64,
            learnable=True,
        )

        # Feature grid sampler
        self.grid_sampler = FeatureGridSampler(
            dims=feature_dims,
            output_dim=hidden_dim,
        )

        # Modulated SIREN
        self.siren = ModulatedSIREN(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            feature_dim=hidden_dim,
            num_layers=num_siren_layers,
            omega_0=omega_0,
        )

        # Refinement network (optional post-processing)
        self.refine = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self,
                coords: torch.Tensor,
                feature_grids: List[torch.Tensor],
                biases: List[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            coords: [B, Q, 3] query coordinates
            feature_grids: List of [B, C_i, H_i, W_i] from aggregator
            biases: Optional scene-level biases

        Returns:
            [B, Q, 3] RGB values
        """
        # Encode coordinates
        coord_features = self.coord_encoder(coords)  # [B, Q, coord_dim]

        # Sample from feature grids
        spatial_features = self.grid_sampler(feature_grids, coords)  # [B, Q, hidden_dim]

        # SIREN with modulation
        rgb = self.siren(coord_features, spatial_features, biases)

        # Optional refinement
        rgb = rgb + 0.1 * self.refine(rgb)  # Residual refinement
        rgb = rgb.clamp(0, 1)

        return rgb


class UpsampleDecoder(nn.Module):
    """
    CNN-based upsampling decoder for efficient full-frame rendering.

    Instead of querying every pixel with SIREN, uses learned upsampling.
    Much faster for full-frame rendering.
    """

    def __init__(self,
                 feature_dims: List[int] = None,
                 hidden_dim: int = 128):
        super().__init__()

        if feature_dims is None:
            feature_dims = [64, 128, 256, 512]

        # Reverse feature pyramid (decode from coarse to fine)
        self.upsample_blocks = nn.ModuleList()

        # Start from coarsest
        in_dim = feature_dims[-1]
        for i in range(len(feature_dims) - 1, -1, -1):
            out_dim = feature_dims[i - 1] if i > 0 else hidden_dim

            block = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
            self.upsample_blocks.append(block)
            in_dim = out_dim + (feature_dims[i - 1] if i > 0 else 0)  # Skip connection

        # Final projection to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature_grids: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feature_grids: List of [B, C_i, H_i, W_i] from coarse to fine

        Returns:
            [B, 3, H, W] RGB image
        """
        # Start from coarsest
        x = feature_grids[-1]

        for i, block in enumerate(self.upsample_blocks):
            x = block(x)

            # Skip connection (if not at finest level yet)
            skip_idx = len(feature_grids) - 2 - i
            if skip_idx >= 0:
                skip = feature_grids[skip_idx]
                # Match spatial size
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)

        return self.to_rgb(x)
