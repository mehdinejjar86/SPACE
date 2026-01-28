# space.py
# SPACE: Spacetime Implicit Neural Representation
# Treats video as a continuous 3D spacetime volume queryable at any (x, y, t)
# Focused on temporal interpolation at original resolution (maintains creator intent)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from .encoder_x import FrameEncoderX
from .aggregator_x import TemporalAggregatorX
from .decoder_x import HybridDecoder, UpsampleDecoder
from .bias_generator import BiasGenerator
from .refinement import RefinementModule
from .nafnet_decoder import NAFNetDecoder


class SPACE(nn.Module):
    """
    SPACE: Spacetime Implicit Neural Representation.

    Treats video as a continuous 3D volume queryable at any (x, y, t).
    Input N frames act as "temporal views" of the scene, analogous to
    how multiple camera angles work in 3D multi-view synthesis.

    Outputs are always at the same resolution as input frames
    (maintains creator intent - no super-resolution).

    Architecture:
        1. ConvNeXt encoder with multi-scale features
        2. Multi-scale temporal aggregation with attention
        3. Feature grid conditioning + bias modulation
        4. Hybrid decoder: FiLM-modulated SIREN + feature sampling
        5. Optional CNN upsampler for fast full-frame rendering

    Usage:
        model = SPACE()

        # Input: N frames at known positions
        frames = torch.randn(B, N, 3, H, W)
        positions = torch.tensor([[0, 0.33, 0.66, 1.0]])

        # Render frame at position 0.5 (same resolution as input)
        output = model.render_frame(frames, positions, target_time=0.5)

        # High-quality mode (slower, more accurate)
        output = model.render_frame(frames, positions, target_time=0.5, mode='hq')

        # Fast mode (CNN upsampler)
        output = model.render_frame(frames, positions, target_time=0.5, mode='fast')
    """

    def __init__(self,
                 # Encoder config
                 encoder_dims: List[int] = None,
                 encoder_depths: List[int] = None,

                 # Aggregator config
                 latent_dim: int = 256,
                 num_heads: int = 8,

                 # Decoder config
                 hidden_dim: int = 256,
                 num_siren_layers: int = 4,
                 omega_0: float = 30.0,

                 # Features
                 use_feature_grids: bool = True,
                 use_fast_decoder: bool = True,
                 use_gradient_checkpointing: bool = False,

                 # Coarse-to-Fine Refinement
                 use_refinement: bool = False,
                 refinement_hidden_dim: int = 64,
                 refinement_num_blocks: int = 4,

                 # NAFNet-style decoder (alternative to fast_decoder)
                 use_nafnet_decoder: bool = False,
                 nafnet_hidden_dim: int = 64,
                 nafnet_num_blocks: List[int] = None,
                 nafnet_use_modulation: bool = True):
        """
        Args:
            encoder_dims: ConvNeXt channel dims per stage [64, 128, 256, 512]
            encoder_depths: ConvNeXt block depths [2, 2, 6, 2]
            latent_dim: Global latent dimension
            num_heads: Attention heads
            hidden_dim: SIREN hidden dimension
            num_siren_layers: SIREN depth
            omega_0: SIREN frequency scaling
            use_feature_grids: Output spatial feature grids from aggregator
            use_fast_decoder: Include CNN upsampler for fast rendering
            use_gradient_checkpointing: Use gradient checkpointing to save memory
            use_refinement: Use coarse-to-fine refinement module
            refinement_hidden_dim: Hidden dimension for refinement network
            refinement_num_blocks: Number of residual blocks in refinement
            use_nafnet_decoder: Use NAFNet-style decoder (replaces fast_decoder)
            nafnet_hidden_dim: NAFNet decoder hidden dimension
            nafnet_num_blocks: NAFNet blocks per level [2, 2, 4, 4]
            nafnet_use_modulation: Apply scene/time-conditioned modulation in NAFNet
        """
        super().__init__()

        if encoder_dims is None:
            encoder_dims = [64, 128, 256, 512]
        if encoder_depths is None:
            encoder_depths = [2, 2, 6, 2]
        if nafnet_num_blocks is None:
            nafnet_num_blocks = [2, 2, 4, 4]

        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim
        self.use_feature_grids = use_feature_grids
        self.use_fast_decoder = use_fast_decoder
        self.use_refinement = use_refinement
        self.use_nafnet_decoder = use_nafnet_decoder
        self.nafnet_use_modulation = nafnet_use_modulation

        # 1. Multi-scale Frame Encoder (ConvNeXt)
        self.encoder = FrameEncoderX(
            latent_dim=latent_dim,
            dims=encoder_dims,
            depths=encoder_depths,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # 2. Multi-scale Temporal Aggregator
        self.aggregator = TemporalAggregatorX(
            dims=encoder_dims,
            latent_dim=latent_dim,
            num_heads=num_heads,
            output_grid=use_feature_grids,
        )

        # 3. Bias Generator (scene-level modulation)
        siren_dims = [hidden_dim] * (num_siren_layers - 1) + [3]
        self.bias_generator = BiasGenerator(
            scene_dim=latent_dim,
            hidden_dims=siren_dims,
        )

        # 4. Hybrid Decoder (coordinate-based SIREN + feature grids)
        self.decoder = HybridDecoder(
            coord_dim=latent_dim,
            hidden_dim=hidden_dim,
            feature_dims=encoder_dims,
            num_siren_layers=num_siren_layers,
            omega_0=omega_0,
        )

        # 5. Fast CNN Decoder (optional)
        if use_fast_decoder:
            self.fast_decoder = UpsampleDecoder(
                feature_dims=encoder_dims,
                hidden_dim=hidden_dim,
            )

        # 6. Coarse-to-Fine Refinement (optional)
        if use_refinement:
            self.refinement = RefinementModule(
                feature_dims=encoder_dims,
                hidden_dim=refinement_hidden_dim,
                num_res_blocks=refinement_num_blocks,
            )

        # 7. NAFNet-style Decoder (optional, alternative to fast_decoder)
        if use_nafnet_decoder:
            self.nafnet_decoder = NAFNetDecoder(
                feature_dims=encoder_dims,
                hidden_dim=nafnet_hidden_dim,
                num_blocks=nafnet_num_blocks,
                mod_dim=latent_dim if nafnet_use_modulation else None,
            )

    def encode(self,
               frames: torch.Tensor,
               frame_times: torch.Tensor,
               query_time: torch.Tensor) -> Dict:
        """
        Encode frames and aggregate for query time.

        Args:
            frames: [B, N, 3, H, W] input frames
            frame_times: [B, N] frame timestamps
            query_time: [B, 1] or [B] target timestamp

        Returns:
            Dict with scene_code, feature_grids, biases
        """
        if query_time.dim() == 1:
            query_time = query_time.unsqueeze(1)

        # Extract multi-scale features and global latents
        multi_scale, frame_latents = self.encoder(frames)

        # Aggregate across time
        scene_code, feature_grids = self.aggregator(
            multi_scale, frame_latents, frame_times, query_time
        )

        # Generate biases
        biases = self.bias_generator(scene_code)

        return {
            'scene_code': scene_code,
            'feature_grids': feature_grids,
            'biases': biases,
            'multi_scale': multi_scale,
        }

    def decode_coords(self,
                      coords: torch.Tensor,
                      encoding: Dict) -> torch.Tensor:
        """
        Decode RGB at query coordinates.

        Args:
            coords: [B, Q, 3] query (x, y, t) coordinates
            encoding: Output from encode()

        Returns:
            [B, Q, 3] RGB values
        """
        return self.decoder(
            coords,
            encoding['feature_grids'],
            encoding['biases'],
        )

    def _normalize_target_time(self, target_time, B: int, device: torch.device) -> torch.Tensor:
        """Normalize target_time to shape [B, 1] on device."""
        if isinstance(target_time, (int, float)):
            return torch.full((B, 1), float(target_time), device=device)
        t = target_time.to(device)
        if t.dim() == 2:
            t = t.squeeze(1)
        return t.view(B, 1)

    def _render_full_from_encoding(self,
                                   encoding: Dict,
                                   H: int,
                                   W: int,
                                   target_time,
                                   mode: str = 'auto') -> torch.Tensor:
        """Render full frame from a precomputed encoding (no re-encode)."""
        B = encoding['scene_code'].shape[0]
        device = encoding['scene_code'].device

        if mode == 'auto':
            if self.use_nafnet_decoder:
                mode = 'nafnet'
            elif H * W > 256 * 256:
                mode = 'fast'
            else:
                mode = 'hq'

        if mode == 'nafnet' and self.use_nafnet_decoder:
            pred = self.nafnet_decoder(encoding['feature_grids'], encoding['scene_code'])
            if pred.shape[2:] != (H, W):
                pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)
            return pred

        if mode == 'fast' and self.use_fast_decoder:
            coarse = self.fast_decoder(encoding['feature_grids'])
            if coarse.shape[2:] != (H, W):
                coarse = F.interpolate(coarse, size=(H, W),
                                       mode='bilinear', align_corners=True)
            if self.use_refinement:
                feature_grids_resized = []
                for feat in encoding['feature_grids']:
                    if feat.shape[2:] != (H, W):
                        feat = F.interpolate(feat, size=(H, W),
                                             mode='bilinear', align_corners=True)
                    feature_grids_resized.append(feat)
                refined = self.refinement(coarse, feature_grids_resized)
                return refined
            return coarse

        # HQ: coordinate-based rendering from encoding
        query_time = self._normalize_target_time(target_time, B, device)
        t_vals = query_time.squeeze(1).view(B, 1).expand(B, H * W)

        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([
            xx.flatten().unsqueeze(0).expand(B, -1),
            yy.flatten().unsqueeze(0).expand(B, -1),
            t_vals,
        ], dim=-1)

        rgb = self.decode_coords(coords, encoding)
        return rgb.view(B, H, W, 3).permute(0, 3, 1, 2)

    def forward(self,
                frames: torch.Tensor,
                frame_times: torch.Tensor,
                query_coords: Optional[torch.Tensor],
                target_time: Optional[torch.Tensor] = None,
                return_full: bool = False,
                full_mode: str = 'auto'):
        """
        Query the spacetime volume at arbitrary coordinates.

        Args:
            frames: [B, N, 3, H, W] input frames
            frame_times: [B, N] frame timestamps
            query_coords: [B, Q, 3] query (x, y, t) coordinates
            target_time: Optional target time for full-frame rendering
            return_full: If True, also return full-frame prediction
            full_mode: 'auto', 'nafnet', 'fast', or 'hq'

        Returns:
            [B, Q, 3] RGB values
        """
        B = frames.shape[0]
        device = frames.device

        if query_coords is None and not return_full:
            raise ValueError("query_coords is required unless return_full=True")

        # Use target_time if provided, otherwise infer from query_coords
        if target_time is not None:
            query_time = self._normalize_target_time(target_time, B, device)
        else:
            query_time = query_coords[:, 0, 2:3]  # Assume same t for all queries

        encoding = self.encode(frames, frame_times, query_time)

        pred_rgb = None
        if query_coords is not None:
            pred_rgb = self.decode_coords(query_coords, encoding)

        if return_full:
            H, W = frames.shape[3], frames.shape[4]
            pred_full = self._render_full_from_encoding(
                encoding, H, W, target_time=query_time.squeeze(1), mode=full_mode
            )
            return (pred_rgb, pred_full) if pred_rgb is not None else pred_full

        return pred_rgb

    def render_frame(self,
                     frames: torch.Tensor,
                     frame_times: torch.Tensor,
                     target_time: float,
                     mode: str = 'auto') -> torch.Tensor:
        """
        Render a complete frame at target timestamp.

        Output resolution matches input frame resolution (no super-resolution).

        Args:
            frames: [B, N, 3, H, W] input frames
            frame_times: [B, N] frame timestamps
            target_time: Target timestamp to render
            mode: 'auto', 'hq' (coordinate-based), or 'fast' (CNN upsampler)

        Returns:
            [B, 3, H, W] rendered frame at input resolution
        """
        H, W = frames.shape[3], frames.shape[4]

        if mode == 'auto':
            # Prefer NAFNet if available; otherwise fast for large, HQ for small
            if self.use_nafnet_decoder:
                mode = 'nafnet'
            elif H * W > 256 * 256:
                mode = 'fast'
            else:
                mode = 'hq'

        if mode == 'nafnet' and self.use_nafnet_decoder:
            return self._render_nafnet(frames, frame_times, target_time)
        if mode == 'fast' and self.use_fast_decoder:
            return self._render_fast(frames, frame_times, target_time)
        return self._render_hq(frames, frame_times, target_time)

    def _render_hq(self,
                   frames: torch.Tensor,
                   frame_times: torch.Tensor,
                   target_time: float) -> torch.Tensor:
        """High-quality rendering using coordinate queries."""
        B = frames.shape[0]
        H, W = frames.shape[3], frames.shape[4]
        device = frames.device

        # Encode
        if isinstance(target_time, (int, float)):
            query_time = torch.full((B, 1), target_time, device=device)
            t_vals = torch.full((B, H * W), target_time, device=device)
        else:
            t = target_time.to(device)
            if t.dim() == 2:
                t = t.squeeze(1)
            query_time = t.unsqueeze(1)
            t_vals = t.view(B, 1).expand(B, H * W)
        encoding = self.encode(frames, frame_times, query_time)

        # Create coordinate grid at input resolution
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        coords = torch.stack([
            xx.flatten().unsqueeze(0).expand(B, -1),
            yy.flatten().unsqueeze(0).expand(B, -1),
            t_vals,
        ], dim=-1)

        # Decode
        rgb = self.decode_coords(coords, encoding)

        return rgb.view(B, H, W, 3).permute(0, 3, 1, 2)

    def _render_nafnet(self,
                       frames: torch.Tensor,
                       frame_times: torch.Tensor,
                       target_time: float) -> torch.Tensor:
        """Full-frame rendering using NAFNet-style decoder."""
        if not self.use_nafnet_decoder:
            return self._render_fast(frames, frame_times, target_time)

        B = frames.shape[0]
        H, W = frames.shape[3], frames.shape[4]
        device = frames.device

        if isinstance(target_time, (int, float)):
            query_time = torch.full((B, 1), target_time, device=device)
        else:
            t = target_time.to(device)
            if t.dim() == 2:
                t = t.squeeze(1)
            query_time = t.unsqueeze(1)

        encoding = self.encode(frames, frame_times, query_time)
        pred = self.nafnet_decoder(encoding['feature_grids'], encoding['scene_code'])

        if pred.shape[2:] != (H, W):
            pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)
        return pred

    def _render_fast(self,
                     frames: torch.Tensor,
                     frame_times: torch.Tensor,
                     target_time: float,
                     return_coarse: bool = False) -> torch.Tensor:
        """Fast rendering using CNN upsampler with optional refinement."""
        if not self.use_fast_decoder:
            return self._render_hq(frames, frame_times, target_time)

        B = frames.shape[0]
        H, W = frames.shape[3], frames.shape[4]
        device = frames.device

        # Encode
        if isinstance(target_time, (int, float)):
            query_time = torch.full((B, 1), target_time, device=device)
        else:
            t = target_time.to(device)
            if t.dim() == 2:
                t = t.squeeze(1)
            query_time = t.unsqueeze(1)
        encoding = self.encode(frames, frame_times, query_time)

        # Use CNN decoder on feature grids (coarse prediction)
        coarse = self.fast_decoder(encoding['feature_grids'])

        # Resize to input resolution
        if coarse.shape[2] != H or coarse.shape[3] != W:
            coarse = F.interpolate(coarse, size=(H, W),
                                   mode='bilinear', align_corners=True)

        # Apply refinement if enabled
        if self.use_refinement:
            # Resize feature grids to match output resolution for refinement
            feature_grids_resized = []
            for feat in encoding['feature_grids']:
                if feat.shape[2:] != (H, W):
                    feat = F.interpolate(feat, size=(H, W),
                                         mode='bilinear', align_corners=True)
                feature_grids_resized.append(feat)

            refined = self.refinement(coarse, feature_grids_resized)

            if return_coarse:
                return refined, coarse
            return refined

        if return_coarse:
            return coarse, coarse
        return coarse

    def render_frame_chunked(self,
                             frames: torch.Tensor,
                             frame_times: torch.Tensor,
                             target_time: float,
                             chunk_size: int = 4096) -> torch.Tensor:
        """
        Memory-efficient HQ rendering with chunked queries.

        For large images where querying all pixels at once would exceed GPU memory.
        Output resolution matches input frame resolution.

        Args:
            frames: [B, N, 3, H, W] input frames
            frame_times: [B, N] frame timestamps
            target_time: Target timestamp to render
            chunk_size: Number of pixels to process at once

        Returns:
            [B, 3, H, W] rendered frame at input resolution
        """
        B = frames.shape[0]
        H, W = frames.shape[3], frames.shape[4]
        device = frames.device

        # Encode once
        query_time = torch.full((B, 1), target_time, device=device)
        encoding = self.encode(frames, frame_times, query_time)

        # Create full grid at input resolution
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()

        # Process in chunks
        rgb_chunks = []
        num_pixels = H * W

        for i in range(0, num_pixels, chunk_size):
            end_i = min(i + chunk_size, num_pixels)

            coords = torch.stack([
                xx_flat[i:end_i].unsqueeze(0).expand(B, -1),
                yy_flat[i:end_i].unsqueeze(0).expand(B, -1),
                torch.full((B, end_i - i), target_time, device=device),
            ], dim=-1)

            rgb_chunk = self.decode_coords(coords, encoding)
            rgb_chunks.append(rgb_chunk)

        rgb = torch.cat(rgb_chunks, dim=1)
        return rgb.view(B, H, W, 3).permute(0, 3, 1, 2)

    # Legacy methods for backwards compatibility
    def render_frame_hq(self,
                        frames: torch.Tensor,
                        frame_times: torch.Tensor,
                        target_time: float,
                        height: int = None,
                        width: int = None) -> torch.Tensor:
        """Legacy method. Use render_frame(mode='hq') instead."""
        return self._render_hq(frames, frame_times, target_time)

    def render_frame_fast(self,
                          frames: torch.Tensor,
                          frame_times: torch.Tensor,
                          target_time: float,
                          height: int = None,
                          width: int = None) -> torch.Tensor:
        """Legacy method. Use render_frame(mode='fast') instead."""
        return self._render_fast(frames, frame_times, target_time)

    def get_param_count(self) -> Dict[str, int]:
        """Return parameter counts for each component."""
        counts = {
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'aggregator': sum(p.numel() for p in self.aggregator.parameters()),
            'bias_generator': sum(p.numel() for p in self.bias_generator.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
        }
        if self.use_fast_decoder:
            counts['fast_decoder'] = sum(p.numel() for p in self.fast_decoder.parameters())
        if self.use_nafnet_decoder:
            counts['nafnet_decoder'] = sum(p.numel() for p in self.nafnet_decoder.parameters())
        if self.use_refinement:
            counts['refinement'] = sum(p.numel() for p in self.refinement.parameters())
        counts['total'] = sum(counts.values())
        return counts

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing."""
        self.encoder.set_gradient_checkpointing(enable)


def build_space(config: str = 'base', **overrides) -> SPACE:
    """
    Build SPACE model from preset config.

    Configs:
        'tiny': ~4M params, fast training/inference
        'base': ~21M params, good balance
        'large': ~45M params, best quality
    """
    configs = {
        'tiny': {
            'encoder_dims': [32, 64, 128, 256],
            'encoder_depths': [1, 1, 3, 1],
            'latent_dim': 128,
            'hidden_dim': 128,
            'num_siren_layers': 3,
            'use_fast_decoder': False,
            'use_nafnet_decoder': True,
        },
        'base': {
            'encoder_dims': [64, 128, 256, 512],
            'encoder_depths': [2, 2, 6, 2],
            'latent_dim': 256,
            'hidden_dim': 256,
            'num_siren_layers': 4,
            'use_fast_decoder': False,
            'use_nafnet_decoder': True,
        },
        'large': {
            'encoder_dims': [96, 192, 384, 768],
            'encoder_depths': [3, 3, 9, 3],
            'latent_dim': 384,
            'hidden_dim': 384,
            'num_siren_layers': 5,
            'use_fast_decoder': False,
            'use_nafnet_decoder': True,
        },
    }

    cfg = configs.get(config, configs['base']).copy()
    cfg.update(overrides)
    return SPACE(**cfg)
