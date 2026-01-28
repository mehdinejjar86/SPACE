# refinement.py
# Coarse-to-Fine Refinement Module for SPACE
# Predicts residuals to refine coarse predictions

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ResidualBlock(nn.Module):
    """Residual block with pre-activation."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class RefinementModule(nn.Module):
    """
    Coarse-to-Fine Refinement Module.

    Takes a coarse prediction and multi-scale features, outputs a refined prediction.
    Uses residual learning: refined = coarse + predicted_residual

    This module:
    1. Encodes the coarse prediction
    2. Fuses with multi-scale features from the encoder
    3. Predicts a residual correction
    4. Adds residual to coarse prediction

    Benefits:
    - Easier to learn (residuals are typically small)
    - Preserves good parts of coarse prediction
    - Can focus on fixing errors in difficult regions
    """

    def __init__(
        self,
        feature_dims: List[int] = None,
        hidden_dim: int = 64,
        num_res_blocks: int = 4,
        use_attention: bool = True,
    ):
        """
        Args:
            feature_dims: Encoder feature dimensions [64, 128, 256, 512]
            hidden_dim: Hidden dimension for refinement network
            num_res_blocks: Number of residual blocks
            use_attention: Use channel attention for feature fusion
        """
        super().__init__()

        if feature_dims is None:
            feature_dims = [64, 128, 256, 512]

        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Encode coarse prediction (RGB -> features)
        self.coarse_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # Feature fusion layers (one per scale)
        self.feature_projections = nn.ModuleList()
        for dim in feature_dims:
            self.feature_projections.append(
                nn.Conv2d(dim, hidden_dim, 1)
            )

        # Channel attention for feature selection
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim * (len(feature_dims) + 1), hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(feature_dims) + 1),
                nn.Softmax(dim=-1),
            )

        # Refinement network
        self.refinement = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        # Output residual prediction
        self.to_residual = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 3, 3, padding=1),
            nn.Tanh(),  # Bound residuals to [-1, 1]
        )

        # Residual scale (learnable, starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        coarse: torch.Tensor,
        feature_grids: List[torch.Tensor],
        return_residual: bool = False,
    ) -> torch.Tensor:
        """
        Refine coarse prediction using multi-scale features.

        Args:
            coarse: [B, 3, H, W] coarse prediction
            feature_grids: List of [B, C_i, H_i, W_i] encoder features
            return_residual: If True, also return the predicted residual

        Returns:
            refined: [B, 3, H, W] refined prediction
            residual: [B, 3, H, W] predicted residual (if return_residual=True)
        """
        B, _, H, W = coarse.shape

        # Encode coarse prediction
        coarse_feat = self.coarse_encoder(coarse)  # [B, hidden_dim, H, W]

        # Project and upsample features to target resolution
        projected_feats = [coarse_feat]
        for i, (feat, proj) in enumerate(zip(feature_grids, self.feature_projections)):
            feat_proj = proj(feat)  # [B, hidden_dim, H_i, W_i]
            if feat_proj.shape[2:] != (H, W):
                feat_proj = F.interpolate(
                    feat_proj, size=(H, W), mode='bilinear', align_corners=True
                )
            projected_feats.append(feat_proj)

        # Fuse features
        if self.use_attention:
            # Stack for attention computation
            stacked = torch.stack(projected_feats, dim=1)  # [B, N, C, H, W]
            attn_input = stacked.mean(dim=[3, 4])  # [B, N, C]
            attn_input = attn_input.view(B, -1)  # [B, N*C]
            attn_weights = self.attention(attn_input)  # [B, N]
            attn_weights = attn_weights.view(B, -1, 1, 1, 1)  # [B, N, 1, 1, 1]
            fused = (stacked * attn_weights).sum(dim=1)  # [B, C, H, W]
        else:
            # Simple averaging
            fused = torch.stack(projected_feats, dim=0).mean(dim=0)

        # Refine
        refined_feat = self.refinement(fused)

        # Predict residual
        residual = self.to_residual(refined_feat)
        residual = residual * self.residual_scale  # Scale residual

        # Add to coarse
        refined = coarse + residual
        refined = refined.clamp(0, 1)  # Ensure valid range

        if return_residual:
            return refined, residual
        return refined


class IterativeRefinement(nn.Module):
    """
    Iterative refinement that applies the refinement module multiple times.

    Can progressively improve predictions through multiple refinement steps.
    """

    def __init__(
        self,
        refinement_module: RefinementModule,
        num_iterations: int = 2,
        share_weights: bool = True,
    ):
        """
        Args:
            refinement_module: Base refinement module
            num_iterations: Number of refinement iterations
            share_weights: If True, use same weights for all iterations
        """
        super().__init__()

        self.num_iterations = num_iterations
        self.share_weights = share_weights

        if share_weights:
            self.refinements = nn.ModuleList([refinement_module])
        else:
            # Create separate modules for each iteration
            self.refinements = nn.ModuleList([
                RefinementModule(
                    feature_dims=refinement_module.feature_dims,
                    hidden_dim=refinement_module.hidden_dim,
                )
                for _ in range(num_iterations)
            ])

    def forward(
        self,
        coarse: torch.Tensor,
        feature_grids: List[torch.Tensor],
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Iteratively refine the coarse prediction.

        Args:
            coarse: [B, 3, H, W] initial coarse prediction
            feature_grids: List of encoder features
            return_intermediates: If True, return all intermediate predictions

        Returns:
            refined: [B, 3, H, W] final refined prediction
            intermediates: List of intermediate predictions (if return_intermediates=True)
        """
        current = coarse
        intermediates = [coarse] if return_intermediates else None

        for i in range(self.num_iterations):
            module = self.refinements[0] if self.share_weights else self.refinements[i]
            current = module(current, feature_grids)
            if return_intermediates:
                intermediates.append(current)

        if return_intermediates:
            return current, intermediates
        return current


class CoarseToFineDecoder(nn.Module):
    """
    Full coarse-to-fine decoder that wraps an existing decoder.

    Pipeline:
    1. Get coarse prediction from base decoder (fast_decoder or SIREN)
    2. Refine using RefinementModule with encoder features
    """

    def __init__(
        self,
        base_decoder: nn.Module,
        feature_dims: List[int] = None,
        hidden_dim: int = 64,
        num_res_blocks: int = 4,
        num_iterations: int = 1,
    ):
        """
        Args:
            base_decoder: Existing decoder (UpsampleDecoder or HybridDecoder)
            feature_dims: Encoder feature dimensions
            hidden_dim: Refinement hidden dimension
            num_res_blocks: Residual blocks in refinement
            num_iterations: Number of refinement iterations
        """
        super().__init__()

        self.base_decoder = base_decoder
        self.refinement = RefinementModule(
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
        )

        self.num_iterations = num_iterations
        if num_iterations > 1:
            self.iterative = IterativeRefinement(
                self.refinement,
                num_iterations=num_iterations,
                share_weights=True,
            )

    def forward(
        self,
        feature_grids: List[torch.Tensor],
        return_coarse: bool = False,
    ) -> torch.Tensor:
        """
        Decode with coarse-to-fine refinement.

        Args:
            feature_grids: List of encoder features
            return_coarse: If True, also return coarse prediction

        Returns:
            refined: [B, 3, H, W] refined output
            coarse: [B, 3, H, W] coarse output (if return_coarse=True)
        """
        # Get coarse prediction from base decoder
        coarse = self.base_decoder(feature_grids)

        # Refine
        if self.num_iterations > 1:
            refined = self.iterative(coarse, feature_grids)
        else:
            refined = self.refinement(coarse, feature_grids)

        if return_coarse:
            return refined, coarse
        return refined
