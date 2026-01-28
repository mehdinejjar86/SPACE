# nafnet_decoder.py
# NAFNet-style decoder for full-image frame interpolation
# Based on "Simple Baselines for Image Restoration" (ECCV 2022)
# Key: No nonlinear activations, uses SimpleGate (multiplication)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class SimpleGate(nn.Module):
    """Simple gate: splits channels and multiplies (no activation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Simplified channel attention without softmax."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.pool(x)
        attn = self.fc(attn)
        return x * attn


class NAFBlock(nn.Module):
    """
    NAFNet block: LayerNorm → Conv → SimpleGate → Conv → ChannelAttn + Skip

    Key differences from standard blocks:
    - No GELU/ReLU, uses SimpleGate (multiplication)
    - Layer normalization instead of batch norm
    - Simplified channel attention
    """

    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()

        hidden = channels * expansion

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden * 2, 1)  # *2 for SimpleGate
        self.conv2 = nn.Conv2d(hidden, hidden * 2, 3, padding=1, groups=hidden)  # Depthwise
        self.gate = SimpleGate()
        self.conv3 = nn.Conv2d(hidden, channels, 1)
        self.ca = SimplifiedChannelAttention(channels)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, hidden * 2, 1)
        self.conv5 = nn.Conv2d(hidden, channels, 1)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Learnable scaling factors
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First block: spatial mixing
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.gate(x)
        x = self.conv2(x)
        x = self.gate(x)
        x = self.conv3(x)
        x = self.ca(x)
        x = self.dropout(x)
        x = residual + x * self.beta

        # Second block: channel mixing
        residual = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.gate(x)
        x = self.conv5(x)
        x = self.dropout(x)
        x = residual + x * self.gamma

        return x


class NAFNetDecoder(nn.Module):
    """
    NAFNet-style decoder for video frame interpolation.

    Takes multi-scale encoder features and produces full-frame output.
    Designed for end-to-end full-image training (not coordinate sampling).

    Architecture:
    - U-Net structure with NAFBlocks
    - Multi-scale feature fusion
    - Direct RGB output
    """

    def __init__(
        self,
        feature_dims: List[int] = None,
        hidden_dim: int = 64,
        num_blocks: List[int] = None,
        dropout: float = 0.0,
        mod_dim: Optional[int] = None,
    ):
        """
        Args:
            feature_dims: Encoder feature dimensions [64, 128, 256, 512]
            hidden_dim: Base hidden dimension
            num_blocks: Number of NAFBlocks at each decoder level [2, 2, 4, 4]
            dropout: Dropout rate
        """
        super().__init__()

        if feature_dims is None:
            feature_dims = [64, 128, 256, 512]
        if num_blocks is None:
            num_blocks = [2, 2, 4, 4]

        self.feature_dims = feature_dims
        num_levels = len(feature_dims)
        self.mod_dim = mod_dim

        if mod_dim is not None:
            self.film = nn.Linear(mod_dim, hidden_dim * 2)

        # Project encoder features to hidden dim
        self.input_projs = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, 1) for dim in feature_dims
        ])

        # Bottleneck (at coarsest level)
        self.bottleneck = nn.Sequential(*[
            NAFBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks[-1])
        ])

        # Decoder levels (coarse to fine)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()

        for i in range(num_levels - 1, 0, -1):
            # Upsample
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim * 4, 1),
                    nn.PixelShuffle(2),
                )
            )

            # Fusion (skip connection)
            self.fusion_blocks.append(
                nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
            )

            # NAFBlocks at this level
            self.decoder_blocks.append(
                nn.Sequential(*[
                    NAFBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks[i - 1])
                ])
            )

        # Final upsampling to full resolution
        self.final_upsample = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(hidden_dim, hidden_dim * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )

        # Output projection
        self.to_rgb = nn.Sequential(
            LayerNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Use xavier for proper FiLM modulation
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feature_grids: List[torch.Tensor], scene_code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            feature_grids: List of [B, C_i, H_i, W_i] from fine to coarse
            scene_code: [B, D] optional time-conditioned code for modulation

        Returns:
            [B, 3, H, W] RGB output at original resolution
        """
        # Project all features to hidden dim
        projected = [proj(feat) for proj, feat in zip(self.input_projs, feature_grids)]

        if self.mod_dim is not None and scene_code is not None:
            gamma_beta = self.film(scene_code)  # [B, 2*hidden_dim]
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            projected = [p * (1 + gamma) + beta for p in projected]

        # Start from coarsest level
        x = projected[-1]
        x = self.bottleneck(x)

        # Decode (coarse to fine)
        for i, (upsample, fusion, blocks) in enumerate(zip(
            self.upsample_blocks, self.fusion_blocks, self.decoder_blocks
        )):
            x = upsample(x)

            # Skip connection from encoder
            skip_idx = len(projected) - 2 - i
            skip = projected[skip_idx]

            # Match spatial size if needed
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            # Fuse
            x = fusion(torch.cat([x, skip], dim=1))
            x = blocks(x)

        # Final upsample to full resolution
        x = self.final_upsample(x)

        # To RGB
        rgb = self.to_rgb(x)
        rgb = torch.sigmoid(rgb)

        return rgb


class NAFNetFrameInterpolator(nn.Module):
    """
    Full NAFNet-based frame interpolator.

    End-to-end trainable on full images. Takes N input frames and
    produces interpolated frame at target time.

    Architecture:
    1. NAFNet encoder (shared across frames)
    2. Temporal fusion (attention-based)
    3. NAFNet decoder → RGB output
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 32,
        num_encoder_blocks: List[int] = None,
        num_decoder_blocks: List[int] = None,
        num_levels: int = 4,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels: Input channels (3 for RGB)
            base_dim: Base channel dimension
            num_encoder_blocks: Blocks per encoder level [2, 2, 4, 8]
            num_decoder_blocks: Blocks per decoder level [2, 2, 4, 4]
            num_levels: Number of resolution levels
            dropout: Dropout rate
        """
        super().__init__()

        if num_encoder_blocks is None:
            num_encoder_blocks = [2, 2, 4, 8]
        if num_decoder_blocks is None:
            num_decoder_blocks = [2, 2, 4, 4]

        dims = [base_dim * (2 ** i) for i in range(num_levels)]

        # Initial projection
        self.intro = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        for i in range(num_levels):
            # NAFBlocks at this level
            self.encoder_blocks.append(
                nn.Sequential(*[
                    NAFBlock(dims[i], dropout=dropout) for _ in range(num_encoder_blocks[i])
                ])
            )

            # Downsample (except last level)
            if i < num_levels - 1:
                self.downsample_blocks.append(
                    nn.Conv2d(dims[i], dims[i + 1], 2, stride=2)
                )

        # Temporal fusion at each scale
        self.temporal_fusions = nn.ModuleList([
            TemporalFusion(dim) for dim in dims
        ])

        # Decoder
        self.decoder = NAFNetDecoder(
            feature_dims=dims,
            hidden_dim=base_dim,
            num_blocks=num_decoder_blocks,
            dropout=dropout,
        )

    def encode_frame(self, frame: torch.Tensor) -> List[torch.Tensor]:
        """Encode a single frame to multi-scale features."""
        features = []
        x = self.intro(frame)

        for i, blocks in enumerate(self.encoder_blocks):
            x = blocks(x)
            features.append(x)

            if i < len(self.downsample_blocks):
                x = self.downsample_blocks[i](x)

        return features

    def forward(
        self,
        frames: torch.Tensor,
        frame_times: torch.Tensor,
        target_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frames: [B, N, 3, H, W] input frames
            frame_times: [B, N] timestamps of input frames
            target_time: [B] or [B, 1] target timestamp

        Returns:
            [B, 3, H, W] interpolated frame
        """
        B, N = frames.shape[:2]

        if target_time.dim() == 2:
            target_time = target_time.squeeze(1)

        # Encode all frames
        all_features = []
        for i in range(N):
            features = self.encode_frame(frames[:, i])
            all_features.append(features)

        # Fuse temporally at each scale
        fused_features = []
        for scale_idx in range(len(all_features[0])):
            scale_features = torch.stack([f[scale_idx] for f in all_features], dim=1)  # [B, N, C, H, W]
            fused = self.temporal_fusions[scale_idx](scale_features, frame_times, target_time)
            fused_features.append(fused)

        # Decode to RGB
        output = self.decoder(fused_features)

        return output


class TemporalFusion(nn.Module):
    """
    Temporal fusion module with time-aware attention.

    Fuses N frame features at a single scale based on target time.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        # Attention
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

        self.norm = LayerNorm2d(channels)

    def forward(
        self,
        features: torch.Tensor,
        frame_times: torch.Tensor,
        target_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, N, C, H, W] features from N frames
            frame_times: [B, N] timestamps
            target_time: [B] target timestamp

        Returns:
            [B, C, H, W] fused features
        """
        B, N, C, H, W = features.shape

        # Compute time distances
        time_dists = (frame_times - target_time.unsqueeze(1)).abs()  # [B, N]

        # Time embeddings
        time_emb = self.time_mlp(time_dists.unsqueeze(-1))  # [B, N, C]

        # Flatten spatial dims for attention
        features_flat = features.view(B, N, C, H * W)  # [B, N, C, HW]

        # Add time embedding
        features_flat = features_flat + time_emb.unsqueeze(-1)

        # Compute attention weights based on time proximity
        # Closer frames get higher weight
        attn_weights = F.softmax(-time_dists * 10, dim=1)  # [B, N]
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]

        # Weighted sum
        fused = (features_flat * attn_weights).sum(dim=1)  # [B, C, HW]
        fused = fused.view(B, C, H, W)

        # Refine with self-attention-like operation
        q = self.query(fused)
        k = self.key(fused)
        v = self.value(fused)

        # Simple channel attention
        attn = (q * k).mean(dim=[2, 3], keepdim=True)
        attn = torch.sigmoid(attn)
        out = v * attn

        fused = fused + self.out(out)
        fused = self.norm(fused)

        return fused


class SIRENGuidance(nn.Module):
    """
    Lightweight SIREN module that generates guidance signals for NAFNet.

    Outputs:
    - Attention map: Where to focus (spatial attention)
    - Residual features: Fine details to add
    """

    def __init__(
        self,
        coord_dim: int = 3,
        hidden_dim: int = 64,
        feature_dim: int = 256,
        num_layers: int = 3,
        omega_0: float = 30.0,
    ):
        super().__init__()
        import math

        self.omega_0 = omega_0
        self.num_layers = num_layers

        # Fourier coordinate encoding
        num_frequencies = 32
        self.register_buffer(
            'frequencies',
            torch.randn(coord_dim, num_frequencies) * 10.0
        )
        self.coord_proj = nn.Linear(num_frequencies * 2, hidden_dim)

        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # SIREN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim * 2 if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.layers.append(nn.Linear(in_dim, out_dim))

        # Output heads
        self.attention_head = nn.Linear(hidden_dim, 1)  # Spatial attention
        self.residual_head = nn.Linear(hidden_dim, 3)   # RGB residual

        self._init_weights()

    def _init_weights(self):
        import math
        for i, layer in enumerate(self.layers):
            if i == 0:
                bound = 1.0 / layer.in_features
            else:
                bound = math.sqrt(6.0 / layer.in_features) / self.omega_0
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)

        # Initialize outputs to near-zero for stable start
        nn.init.zeros_(self.attention_head.weight)
        nn.init.zeros_(self.attention_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def encode_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Fourier encode coordinates."""
        proj = coords @ self.frequencies
        encoded = torch.cat([torch.sin(2 * 3.14159 * proj),
                             torch.cos(2 * 3.14159 * proj)], dim=-1)
        return self.coord_proj(encoded)

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        scene_code: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: [B, Q, 3] query coordinates (x, y, t)
            features: [B, Q, feature_dim] sampled spatial features
            scene_code: Optional [B, feature_dim] scene conditioning (unused, for API compat)

        Returns:
            attention: [B, Q, 1] attention weights
            residual: [B, Q, 3] RGB residual
        """
        # Encode coordinates
        coord_feat = self.encode_coords(coords)  # [B, Q, hidden_dim]

        # Project features
        feat_proj = self.feature_proj(features)  # [B, Q, hidden_dim]

        # Concatenate
        x = torch.cat([coord_feat, feat_proj], dim=-1)  # [B, Q, hidden_dim*2]

        # SIREN layers (no external biases - keeps SIREN guidance independent)
        for layer in self.layers:
            x = layer(x)
            x = torch.sin(self.omega_0 * x)

        # Output heads
        attention = torch.sigmoid(self.attention_head(x))  # [B, Q, 1]
        residual = self.residual_head(x) * 0.1  # Small residual, [B, Q, 3]

        return attention, residual


class HybridNAFNetSIREN(nn.Module):
    """
    Hybrid decoder combining NAFNet (full-frame) with SIREN (coordinate-based guidance).

    Architecture:
        1. NAFNet produces full-frame base output and intermediate features
        2. SIREN provides coordinate-aware guidance:
           - Attention: Where NAFNet should focus
           - Residual: Fine details to add
        3. Final output = NAFNet_base * attention + SIREN_residual

    Benefits:
        - NAFNet: Fast, efficient, good at structure
        - SIREN: Continuous, good at fine details, temporally consistent
        - Hybrid: Best of both worlds
    """

    def __init__(
        self,
        feature_dims: List[int] = None,
        hidden_dim: int = 64,
        num_blocks: List[int] = None,
        latent_dim: int = 256,
        siren_hidden_dim: int = 64,
        siren_num_layers: int = 3,
        omega_0: float = 30.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            feature_dims: Encoder feature dimensions [64, 128, 256, 512]
            hidden_dim: NAFNet hidden dimension
            num_blocks: NAFNet blocks per level [2, 2, 4, 4]
            latent_dim: Scene code dimension (for FiLM modulation)
            siren_hidden_dim: SIREN hidden dimension
            siren_num_layers: SIREN depth
            omega_0: SIREN frequency scaling
            dropout: Dropout rate
        """
        super().__init__()

        if feature_dims is None:
            feature_dims = [64, 128, 256, 512]
        if num_blocks is None:
            num_blocks = [2, 2, 4, 4]

        self.feature_dims = feature_dims
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # NAFNet decoder (produces full-frame output + features)
        self.nafnet = NAFNetDecoder(
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            mod_dim=latent_dim,
        )

        # Feature sampler for SIREN (samples from encoder features)
        total_feat_dim = sum(feature_dims)
        self.feat_projs = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, 1) for dim in feature_dims
        ])
        self.feat_fusion = nn.Linear(hidden_dim * len(feature_dims), latent_dim)

        # SIREN guidance module
        self.siren_guidance = SIRENGuidance(
            coord_dim=3,
            hidden_dim=siren_hidden_dim,
            feature_dim=latent_dim,
            num_layers=siren_num_layers,
            omega_0=omega_0,
        )

        # Learnable blend factor (starts favoring NAFNet)
        self.blend_alpha = nn.Parameter(torch.tensor(0.1))

    def sample_features(
        self,
        feature_grids: List[torch.Tensor],
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample and fuse features from multi-scale grids at query coordinates.

        Args:
            feature_grids: List of [B, C_i, H_i, W_i]
            coords: [B, Q, 3] query coordinates

        Returns:
            [B, Q, latent_dim] sampled features
        """
        B, Q, _ = coords.shape
        xy = coords[:, :, :2]  # Spatial coords
        grid = xy.view(B, Q, 1, 2)

        sampled = []
        for feat, proj in zip(feature_grids, self.feat_projs):
            # Project features
            feat_proj = proj(feat)  # [B, hidden_dim, H, W]
            # Sample at coordinates
            s = F.grid_sample(feat_proj, grid, mode='bilinear',
                              padding_mode='border', align_corners=True)
            s = s.squeeze(-1).permute(0, 2, 1)  # [B, Q, hidden_dim]
            sampled.append(s)

        # Concatenate and fuse
        concat = torch.cat(sampled, dim=-1)  # [B, Q, hidden_dim * num_scales]
        fused = self.feat_fusion(concat)  # [B, Q, latent_dim]

        return fused

    def forward(
        self,
        feature_grids: List[torch.Tensor],
        scene_code: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Hybrid forward pass.

        Args:
            feature_grids: List of [B, C_i, H_i, W_i] encoder features
            scene_code: [B, latent_dim] scene/time conditioning
            coords: Optional [B, Q, 3] query coordinates for SIREN guidance
                    If None, creates a dense grid at output resolution
            biases: Optional scene-level biases for SIREN

        Returns:
            [B, 3, H, W] RGB output
        """
        B = feature_grids[0].shape[0]
        device = feature_grids[0].device

        # 1. NAFNet: full-frame base output
        nafnet_out = self.nafnet(feature_grids, scene_code)  # [B, 3, H_naf, W_naf]
        H_naf, W_naf = nafnet_out.shape[2:]

        # 2. Determine output spatial dimensions
        if coords is None:
            # Use NAFNet output size
            H, W = H_naf, W_naf
            y = torch.linspace(-1, 1, H, device=device)
            x = torch.linspace(-1, 1, W, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([
                xx.flatten(),
                yy.flatten(),
                torch.zeros_like(xx.flatten()),  # t=0 placeholder
            ], dim=-1).unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 3]
        else:
            # Infer H, W from coords (assuming square grid or provided shape)
            Q = coords.shape[1]
            H = W = int(Q ** 0.5)
            if H * W != Q:
                # Non-square, try to infer from NAFNet output aspect ratio
                aspect = W_naf / H_naf
                H = int((Q / aspect) ** 0.5)
                W = Q // H
                if H * W != Q:
                    # Fallback: just use NAFNet output size and sample
                    H, W = H_naf, W_naf

        # 3. Sample features at coordinates
        sampled_feats = self.sample_features(feature_grids, coords)  # [B, Q, latent_dim]

        # 4. SIREN guidance (uses features, not biases - keeps it independent)
        attention, residual = self.siren_guidance(coords, sampled_feats)
        # attention: [B, Q, 1], residual: [B, Q, 3]

        Q = coords.shape[1]

        # 5. Reshape attention and residual to spatial (if Q == H*W)
        if Q == H * W:
            attention_map = attention.view(B, H, W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]
            residual_map = residual.view(B, H, W, 3).permute(0, 3, 1, 2)    # [B, 3, H, W]

            # Resize NAFNet output if needed
            if (H_naf, W_naf) != (H, W):
                nafnet_out = F.interpolate(nafnet_out, size=(H, W), mode='bilinear', align_corners=True)

            # 6. Blend: NAFNet modulated by attention + residual
            alpha = torch.sigmoid(self.blend_alpha)
            output = nafnet_out * (1 + alpha * (attention_map - 0.5)) + alpha * residual_map
            output = output.clamp(0, 1)
        else:
            # coords don't form a regular grid - sample NAFNet at coords
            xy = coords[:, :, :2]
            grid = xy.view(B, Q, 1, 2)
            nafnet_sampled = F.grid_sample(nafnet_out, grid, mode='bilinear',
                                           padding_mode='border', align_corners=True)
            nafnet_sampled = nafnet_sampled.squeeze(-1).permute(0, 2, 1)  # [B, Q, 3]

            # Blend in coordinate space
            alpha = torch.sigmoid(self.blend_alpha)
            output_coords = nafnet_sampled * (1 + alpha * (attention - 0.5)) + alpha * residual
            output_coords = output_coords.clamp(0, 1)

            # Reshape to spatial (use NAFNet output size as fallback)
            output = output_coords.view(B, H_naf, W_naf, 3).permute(0, 3, 1, 2)

        return output

    def forward_coords(
        self,
        feature_grids: List[torch.Tensor],
        scene_code: torch.Tensor,
        coords: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Coordinate-based forward for training with sampled coordinates.

        Args:
            feature_grids: List of [B, C_i, H_i, W_i]
            scene_code: [B, latent_dim]
            coords: [B, Q, 3] sampled coordinates
            biases: Optional biases

        Returns:
            [B, Q, 3] RGB at queried coordinates
        """
        B, Q, _ = coords.shape
        device = coords.device

        # Get NAFNet full output
        nafnet_out = self.nafnet(feature_grids, scene_code)  # [B, 3, H, W]

        # Sample NAFNet output at coordinates
        xy = coords[:, :, :2]
        grid = xy.view(B, Q, 1, 2)
        nafnet_sampled = F.grid_sample(nafnet_out, grid, mode='bilinear',
                                       padding_mode='border', align_corners=True)
        nafnet_sampled = nafnet_sampled.squeeze(-1).permute(0, 2, 1)  # [B, Q, 3]

        # Sample features and get SIREN guidance
        sampled_feats = self.sample_features(feature_grids, coords)
        attention, residual = self.siren_guidance(coords, sampled_feats)

        # Blend
        alpha = torch.sigmoid(self.blend_alpha)
        output = nafnet_sampled * (1 + alpha * (attention - 0.5)) + alpha * residual
        output = output.clamp(0, 1)

        return output
