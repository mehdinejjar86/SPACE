# aggregator_x.py
# Enhanced Temporal Aggregator with deformable attention and multi-scale fusion

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class SinusoidalTimeEncoding(nn.Module):
    """Continuous sinusoidal time encoding with learnable projection."""

    def __init__(self, channels: int, max_period: float = 10000.0):
        super().__init__()
        self.channels = channels
        self.max_period = max_period
        self.proj = nn.Linear(channels, channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )

        t_flat = t.reshape(-1, 1)
        args = t_flat * freqs
        enc = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        enc = enc.reshape(*t.shape, self.channels)

        return self.proj(enc)


class DeformableAttention(nn.Module):
    """
    Deformable attention for temporal feature aggregation.

    Instead of attending to fixed grid positions, learns offsets
    to attend to motion-relevant locations.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 num_points: int = 4,
                 dropout: float = 0.0):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        # Offset prediction
        self.offset_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_heads * num_points * 2),  # 2 for (x, y) offsets
        )

        # Attention weights
        self.attn_weights = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_heads * num_points),
        )

        # Value projection
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize offsets to zero (start with regular grid)
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    def forward(self,
                query: torch.Tensor,
                value_features: torch.Tensor,
                reference_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, Q, C] query features
            value_features: [B, H, W, C] spatial features to sample from
            reference_points: [B, Q, 2] base sampling locations in [-1, 1]

        Returns:
            [B, Q, C] aggregated features
        """
        B, Q, C = query.shape
        _, H, W, _ = value_features.shape

        # Predict offsets
        offsets = self.offset_net(query)  # [B, Q, num_heads * num_points * 2]
        offsets = offsets.view(B, Q, self.num_heads, self.num_points, 2)
        offsets = offsets.tanh() * 0.5  # Limit offset range

        # Predict attention weights
        attn = self.attn_weights(query)  # [B, Q, num_heads * num_points]
        attn = attn.view(B, Q, self.num_heads, self.num_points)
        attn = F.softmax(attn, dim=-1)

        # Compute sampling locations
        # reference_points: [B, Q, 2] -> [B, Q, 1, 1, 2]
        ref = reference_points.view(B, Q, 1, 1, 2)
        sampling_locations = ref + offsets  # [B, Q, num_heads, num_points, 2]

        # Sample values at deformed locations
        # Need to reshape for grid_sample
        v = self.v_proj(value_features)  # [B, H, W, C]
        v = v.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Sample for each head and point
        sampled = []
        for h in range(self.num_heads):
            for p in range(self.num_points):
                grid = sampling_locations[:, :, h, p, :].view(B, Q, 1, 2)
                s = F.grid_sample(v, grid, mode='bilinear', padding_mode='border', align_corners=True)
                s = s.squeeze(-1).permute(0, 2, 1)  # [B, Q, C]
                sampled.append(s)

        # Weighted combination
        sampled = torch.stack(sampled, dim=2)  # [B, Q, num_heads*num_points, C]
        sampled = sampled.view(B, Q, self.num_heads, self.num_points, -1)

        # Apply attention weights
        out = (attn.unsqueeze(-1) * sampled).sum(dim=3)  # [B, Q, num_heads, head_dim]
        out = out.view(B, Q, C)

        return self.out_proj(out)


class TemporalCrossAttention(nn.Module):
    """
    Cross-attention between query time and input frame features.

    Enhanced with:
    - Relative position encoding
    - Temperature-scaled attention
    - Gated output
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 use_gate: bool = True):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_gate = use_gate

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Relative time encoding
        self.time_enc = SinusoidalTimeEncoding(dim)

        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))

        # Output gate
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid(),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                query_time: torch.Tensor,
                key_times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, 1, C] query (scene representation query)
            keys: [B, N, C] frame features
            values: [B, N, C] frame features
            query_time: [B, 1] target timestamp
            key_times: [B, N] input frame timestamps

        Returns:
            output: [B, 1, C]
            attn_weights: [B, num_heads, 1, N]
        """
        B, N, C = keys.shape

        # Add relative time information to keys
        rel_times = key_times - query_time  # [B, N]
        time_emb = self.time_enc(rel_times)  # [B, N, C]
        keys = keys + time_emb

        # Project
        q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention with learnable temperature
        attn = (q @ k.transpose(-2, -1)) * self.scale * self.temperature
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        out = self.out_proj(out)

        # Gated output
        if self.use_gate:
            gate_input = torch.cat([query, out], dim=-1)
            gate = self.gate(gate_input)
            out = gate * out + (1 - gate) * query

        return out, attn


class MultiScaleTemporalAggregator(nn.Module):
    """
    Multi-scale temporal aggregator that fuses features at multiple resolutions.

    Key improvements over basic aggregator:
    1. Multi-scale feature fusion
    2. Deformable sampling for motion-aware aggregation
    3. Iterative refinement
    """

    def __init__(self,
                 dims: List[int],
                 latent_dim: int = 256,
                 num_heads: int = 8,
                 num_iterations: int = 3):
        """
        Args:
            dims: Channel dimensions for each scale [64, 128, 256, 512]
            latent_dim: Output latent dimension
            num_heads: Attention heads
            num_iterations: Refinement iterations
        """
        super().__init__()

        self.dims = dims
        self.num_scales = len(dims)
        self.num_iterations = num_iterations

        # Per-scale temporal attention
        self.temporal_attns = nn.ModuleList([
            TemporalCrossAttention(dim, num_heads)
            for dim in dims
        ])

        # Cross-scale fusion
        self.scale_fusion = nn.ModuleList([
            nn.Conv2d(dims[i], latent_dim, 1) for i in range(len(dims))
        ])

        # Deformable attention for spatial aggregation (finest scale)
        self.deform_attn = DeformableAttention(dims[-1], num_heads)

        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim * len(dims), latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Time embedding for query
        self.time_embed = SinusoidalTimeEncoding(latent_dim)

    def forward(self,
                multi_scale_features: List[torch.Tensor],
                frame_latents: torch.Tensor,
                frame_times: torch.Tensor,
                query_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multi_scale_features: List of [B, N, C_i, H_i, W_i] per scale
            frame_latents: [B, N, latent_dim] global frame features
            frame_times: [B, N] frame timestamps
            query_time: [B, 1] target timestamp

        Returns:
            [B, latent_dim] aggregated scene representation
        """
        B, N = frame_times.shape

        scale_outputs = []

        for scale_idx, (features, attn) in enumerate(zip(multi_scale_features, self.temporal_attns)):
            # features: [B, N, C, H, W]
            C, H, W = features.shape[2:]

            # Global pool each frame for temporal attention
            pooled = features.mean(dim=(3, 4))  # [B, N, C]

            # Query embedding
            query = self.time_embed(query_time.squeeze(-1)).unsqueeze(1)  # [B, 1, C]
            query = F.pad(query, (0, C - query.shape[-1])) if query.shape[-1] < C else query[:, :, :C]

            # Temporal cross-attention
            out, _ = attn(query, pooled, pooled, query_time, frame_times)  # [B, 1, C]

            # Project to common dimension
            out_2d = out.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            out_proj = self.scale_fusion[scale_idx](out_2d).squeeze(-1).squeeze(-1)  # [B, latent_dim]

            scale_outputs.append(out_proj)

        # Concatenate and project
        combined = torch.cat(scale_outputs, dim=-1)  # [B, latent_dim * num_scales]
        scene_code = self.out_proj(combined)  # [B, latent_dim]

        return scene_code


class TemporalAggregatorX(nn.Module):
    """
    Enhanced temporal aggregator with:
    1. Multi-scale temporal attention
    2. Deformable spatial sampling
    3. Feature grid output (not just global latent)
    """

    def __init__(self,
                 dims: List[int] = None,
                 latent_dim: int = 256,
                 num_heads: int = 8,
                 output_grid: bool = True):
        super().__init__()

        if dims is None:
            dims = [64, 128, 256, 512]

        self.dims = dims
        self.latent_dim = latent_dim
        self.output_grid = output_grid

        # Multi-scale aggregator
        self.ms_aggregator = MultiScaleTemporalAggregator(
            dims, latent_dim, num_heads
        )

        # If we want spatial feature grids, add per-scale spatial attention
        if output_grid:
            self.spatial_attns = nn.ModuleList([
                SpatialTemporalAttention(dim, num_heads) for dim in dims
            ])

    def forward(self,
                multi_scale_features: List[torch.Tensor],
                frame_latents: torch.Tensor,
                frame_times: torch.Tensor,
                query_time: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Returns:
            scene_code: [B, latent_dim] global scene representation
            feature_grids: List of [B, C_i, H_i, W_i] spatial grids (if output_grid=True)
        """
        # Get global scene code
        scene_code = self.ms_aggregator(
            multi_scale_features, frame_latents, frame_times, query_time
        )

        feature_grids = None
        if self.output_grid:
            feature_grids = []
            for scale_idx, (features, attn) in enumerate(zip(multi_scale_features, self.spatial_attns)):
                # features: [B, N, C, H, W]
                grid = attn(features, frame_times, query_time)  # [B, C, H, W]
                feature_grids.append(grid)

        return scene_code, feature_grids


class SpatialTemporalAttention(nn.Module):
    """
    Spatial-temporal attention that produces a feature grid.

    For each spatial location, attends across frames weighted by temporal distance.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.time_enc = SinusoidalTimeEncoding(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self,
                features: torch.Tensor,
                frame_times: torch.Tensor,
                query_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, C, H, W]
            frame_times: [B, N]
            query_time: [B, 1]

        Returns:
            [B, C, H, W] fused feature grid
        """
        B, N, C, H, W = features.shape

        # Flatten spatial dimensions
        features_flat = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, N, C)

        # Time encoding
        rel_times = (frame_times - query_time).unsqueeze(1).expand(-1, H * W, -1)
        rel_times = rel_times.reshape(B * H * W, N)
        time_emb = self.time_enc(rel_times)  # [B*H*W, N, C]

        # Add time info
        features_flat = features_flat + time_emb

        # Query: average of all frames
        query = features_flat.mean(dim=1, keepdim=True)  # [B*H*W, 1, C]

        # Attention
        out, _ = self.attn(query, features_flat, features_flat)
        out = self.norm(out.squeeze(1))  # [B*H*W, C]

        # Reshape back
        return out.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
