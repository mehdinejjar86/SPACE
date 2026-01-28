#!/usr/bin/env python3
# train_ddp.py
# Distributed Data Parallel training script for SPACE
# Features:
#   - Multi-GPU training with DDP
#   - Dual-dataset validation (Vimeo + X4K)
#   - Best model saved based on average PSNR

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import amp
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from model import SPACE, build_space, SPACELoss, AdvancedSPACELoss, HybridLoss
from data import (
    VimeoTriplet, X4K1000Dataset, vimeo_collate, x4k_collate,
    DistributedPureBatchSampler, PureBatchSampler,
    X4KBatchSampler, DistributedX4KBatchSampler,
    X4KTestDataset, x4k_test_collate,
    X4KSequenceDataset, x4k_sequence_collate,
)
from config import Config, get_default_config, get_fast_config, get_no_uncertainty_config, get_mixed_config


def setup_ddp(rank: int, world_size: int):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP process group."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def set_seed(seed: int, rank: int = 0):
    """Set random seed for reproducibility."""
    seed = seed + rank  # Different seed per rank for diversity
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config) -> SPACE:
    """Create SPACE model from config."""
    model_config = config.model
    if model_config.preset:
        return build_space(
            model_config.preset,
            use_feature_grids=model_config.use_feature_grids,
            use_fast_decoder=model_config.use_fast_decoder,
            use_refinement=model_config.use_refinement,
            refinement_hidden_dim=model_config.refinement_hidden_dim,
            refinement_num_blocks=model_config.refinement_num_blocks,
            use_nafnet_decoder=model_config.use_nafnet_decoder,
            nafnet_hidden_dim=model_config.nafnet_hidden_dim,
            nafnet_num_blocks=model_config.nafnet_num_blocks,
            nafnet_use_modulation=model_config.nafnet_use_modulation,
            use_hybrid_decoder=getattr(model_config, 'use_hybrid_decoder', True),
            hybrid_siren_hidden_dim=getattr(model_config, 'hybrid_siren_hidden_dim', 64),
            hybrid_siren_layers=getattr(model_config, 'hybrid_siren_layers', 3),
        )
    return SPACE(
        encoder_dims=model_config.encoder_dims,
        encoder_depths=model_config.encoder_depths,
        latent_dim=model_config.latent_dim,
        num_heads=model_config.num_heads,
        hidden_dim=model_config.hidden_dim,
        num_siren_layers=model_config.num_siren_layers,
        omega_0=model_config.omega_0,
        use_feature_grids=model_config.use_feature_grids,
        use_fast_decoder=model_config.use_fast_decoder,
        use_refinement=model_config.use_refinement,
        refinement_hidden_dim=model_config.refinement_hidden_dim,
        refinement_num_blocks=model_config.refinement_num_blocks,
        use_nafnet_decoder=model_config.use_nafnet_decoder,
        nafnet_hidden_dim=model_config.nafnet_hidden_dim,
        nafnet_num_blocks=model_config.nafnet_num_blocks,
        nafnet_use_modulation=model_config.nafnet_use_modulation,
        use_hybrid_decoder=getattr(model_config, 'use_hybrid_decoder', True),
        hybrid_siren_hidden_dim=getattr(model_config, 'hybrid_siren_hidden_dim', 64),
        hybrid_siren_layers=getattr(model_config, 'hybrid_siren_layers', 3),
    )


def create_validation_loaders(config, rank: int, world_size: int):
    """
    Create separate validation dataloaders for Vimeo and X4K.

    Automatically derives validation paths from training paths:
    - Vimeo: Same root, uses tri_testlist.txt (split="test")
    - X4K: Parent of x4k_root/train -> uses test/ subdirectory

    Returns:
        dict: {'vimeo': loader, 'x4k': loader}
    """
    loaders = {}

    # Vimeo validation - uses same root as training, split="test" reads tri_testlist.txt
    vimeo_val_root = config.data.vimeo_val_root or config.data.vimeo_root
    if vimeo_val_root:
        vimeo_val_path = Path(vimeo_val_root)
        # Check if tri_testlist.txt exists
        if (vimeo_val_path / "tri_testlist.txt").exists():
            if is_main_process():
                print(f"  Vimeo validation: {vimeo_val_path}")
            vimeo_val = VimeoTriplet(
                root=str(vimeo_val_path),
                split="test",
                mode="interp",  # Standard interpolation for validation
                crop_size=None,  # No crop for validation
                aug_flip=False,
            )

            if world_size > 1:
                sampler = DistributedSampler(vimeo_val, shuffle=False)
            else:
                sampler = None

            loaders['vimeo'] = DataLoader(
                vimeo_val,
                batch_size=config.train.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                collate_fn=vimeo_collate,
            )
            if is_main_process():
                print(f"    {len(vimeo_val)} samples")
        elif is_main_process():
            print(f"  Vimeo validation: skipped (no tri_testlist.txt at {vimeo_val_path})")

    # X4K validation - uses different structure (Type1/Type2/Type3 with 33 frames each)
    # Uses cascaded 8x interpolation: N=2 → N=3 → N=4
    # Auto-derive: if x4k_root is .../train, look for sibling test/
    x4k_val_root = config.data.x4k_val_root
    if not x4k_val_root and config.data.x4k_root:
        resolved = resolve_x4k_root(config.data.x4k_root)
        # Check for test/ subdirectory (X4K test has Type1/Type2/Type3 inside)
        test_path = Path(resolved) / "test"
        if test_path.exists() and (test_path / "Type1").exists():
            x4k_val_root = str(test_path)

    if x4k_val_root:
        x4k_val_path = Path(x4k_val_root)
        # X4K test has Type1/Type2/Type3 structure with 33 frames per sequence
        if (x4k_val_path / "Type1").exists():
            if is_main_process():
                print(f"  X4K validation: {x4k_val_path} (cascaded 8x interpolation)")

            # Create loaders for both 2K and 4K evaluation
            for scale in ['2k', '4k']:
                x4k_val = X4KSequenceDataset(
                    root=str(x4k_val_path),
                    scale=scale,
                    target_indices=[4, 8, 12, 16, 20, 24, 28],  # 8x interpolation
                )

                if world_size > 1:
                    sampler = DistributedSampler(x4k_val, shuffle=False)
                else:
                    sampler = None

                # Batch size of 1 for cascaded evaluation (full resolution)
                loaders[f'x4k_{scale}'] = DataLoader(
                    x4k_val,
                    batch_size=1,  # Process one sequence at a time for cascaded eval
                    shuffle=False,
                    sampler=sampler,
                    num_workers=max(1, config.data.num_workers // 2),
                    pin_memory=config.data.pin_memory,
                    collate_fn=x4k_sequence_collate,
                )
                if is_main_process():
                    print(f"    X4K-{scale.upper()}: {len(x4k_val)} sequences × 7 targets")
        elif is_main_process():
            print(f"  X4K validation: skipped (no Type1/ at {x4k_val_path})")

    return loaders


def mixed_collate_fn(batch):
    """
    Collate function for mixed dataset batches.
    Since PureBatchSampler ensures pure batches, all samples have same N.
    """
    frames = torch.stack([b[0] for b in batch], dim=0)
    anchor_times = torch.stack([b[1] for b in batch], dim=0)
    target_time = torch.stack([b[2] for b in batch], dim=0)
    target = torch.stack([b[3] for b in batch], dim=0)
    return frames, anchor_times, target_time, target


class MixedDataLoaders:
    """
    Wrapper that alternates between multiple dataloaders based on ratios.

    For mixed Vimeo + X4K training where:
    - Vimeo batches have N=2 anchor frames
    - X4K batches have variable N (2, 3, or 4) depending on step

    Each batch is pure (from one dataset only with same N).
    """

    def __init__(self, vimeo_loader, x4k_loader, ratios=None):
        """
        Args:
            vimeo_loader: DataLoader for Vimeo dataset
            x4k_loader: DataLoader for X4K dataset
            ratios: [vimeo_ratio, x4k_ratio] e.g., [0.7, 0.3]
        """
        self.vimeo_loader = vimeo_loader
        self.x4k_loader = x4k_loader
        self.ratios = ratios or [0.7, 0.3]

        # Calculate total batches and batch assignment
        self.vimeo_batches = len(vimeo_loader)
        self.x4k_batches = len(x4k_loader)
        self.total_batches = self.vimeo_batches + self.x4k_batches

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        """Yield batches alternating between datasets based on ratios."""
        vimeo_iter = iter(self.vimeo_loader)
        x4k_iter = iter(self.x4k_loader)

        vimeo_prob = self.ratios[0]

        vimeo_remaining = self.vimeo_batches
        x4k_remaining = self.x4k_batches

        while vimeo_remaining > 0 or x4k_remaining > 0:
            # Decide which dataset to sample from
            if vimeo_remaining == 0:
                use_vimeo = False
            elif x4k_remaining == 0:
                use_vimeo = True
            else:
                use_vimeo = random.random() < vimeo_prob

            try:
                if use_vimeo:
                    batch = next(vimeo_iter)
                    vimeo_remaining -= 1
                    yield (*batch, "vimeo")
                else:
                    batch = next(x4k_iter)
                    x4k_remaining -= 1
                    yield (*batch, "x4k")
            except StopIteration:
                # One iterator exhausted, continue with the other
                continue


def resolve_x4k_root(x4k_path: str) -> str:
    """
    Resolve X4K root path.

    X4K dataset expects structure: root/train/ and root/test/
    If user provides path ending in 'train', use parent directory.
    """
    path = Path(x4k_path)
    if path.name == "train" and path.parent.exists():
        return str(path.parent)
    return str(path)


def _select_x4k_pairs_for_n(data_config, target_n: int):
    pairs = list(zip(data_config.x4k_steps, data_config.x4k_n_frames))
    selected = [(s, n) for s, n in pairs if n == target_n]
    if not selected:
        selected = [pairs[0]]
    steps = [s for s, _ in selected]
    n_frames = [n for _, n in selected]
    return steps, n_frames


def get_curriculum_settings(config, epoch: int):
    """
    Curriculum schedule (first fraction of training):
      0) Vimeo only
      1) X4K only, N=4
      2) X4K only, N=3
      3) X4K only, N=2
    Then switch to full mixed training.
    """
    data_config = config.data
    train_config = config.train

    if not getattr(train_config, 'curriculum_enabled', False):
        mode = "mixed" if data_config.use_mixed_training else "vimeo_only"
        return {
            'name': mode,
            'mode': mode,
            'x4k_steps': data_config.x4k_steps,
            'x4k_n_frames': data_config.x4k_n_frames,
            'dataset_ratios': data_config.dataset_ratios,
        }

    total_epochs = max(1, train_config.epochs)
    cur_epochs = max(1, int(total_epochs * getattr(train_config, 'curriculum_fraction', 0.2)))

    if epoch >= cur_epochs:
        mode = "mixed" if data_config.use_mixed_training else "vimeo_only"
        return {
            'name': 'mixed_full' if mode == "mixed" else 'vimeo_only',
            'mode': mode,
            'x4k_steps': data_config.x4k_steps,
            'x4k_n_frames': data_config.x4k_n_frames,
            'dataset_ratios': data_config.dataset_ratios,
        }

    stage_len = max(1, cur_epochs // 4)
    stage = min(3, epoch // stage_len)

    if stage == 0:
        return {
            'name': 'vimeo_only',
            'mode': 'vimeo_only',
            'x4k_steps': data_config.x4k_steps,
            'x4k_n_frames': data_config.x4k_n_frames,
            'dataset_ratios': data_config.dataset_ratios,
        }

    target_n = 4 if stage == 1 else 3 if stage == 2 else 2
    steps, n_frames = _select_x4k_pairs_for_n(data_config, target_n)
    return {
        'name': f'x4k_only_n{target_n}',
        'mode': 'x4k_only',
        'x4k_steps': steps,
        'x4k_n_frames': n_frames,
        'dataset_ratios': data_config.dataset_ratios,
    }


def create_train_loader(config, rank: int, world_size: int, mode: str = None,
                        x4k_steps=None, x4k_n_frames=None, dataset_ratios=None):
    """
    Create training dataloader with DDP support.

    Supports:
    - Vimeo-only training (N=2)
    - X4K-only training (variable N per step)
    - Mixed Vimeo + X4K training using alternating loaders
    """
    data_config = config.data
    train_config = config.train
    mode = mode or ("mixed" if data_config.use_mixed_training else "vimeo_only")
    x4k_steps = x4k_steps if x4k_steps is not None else data_config.x4k_steps
    x4k_n_frames = x4k_n_frames if x4k_n_frames is not None else data_config.x4k_n_frames
    dataset_ratios = dataset_ratios if dataset_ratios is not None else data_config.dataset_ratios

    if mode == "x4k_only":
        # X4K-only training (variable N per step)
        if is_main_process():
            print("  Train (X4K only)...")

        x4k_root = resolve_x4k_root(data_config.x4k_root)
        x4k = X4K1000Dataset(
            root=x4k_root,
            split="train",
            steps=x4k_steps,
            n_frames=x4k_n_frames,
            crop_size=data_config.x4k_crop_size,
            aug_flip=True,
        )

        if world_size > 1:
            x4k_sampler = DistributedX4KBatchSampler(
                x4k,
                batch_size=train_config.batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            x4k_sampler = X4KBatchSampler(
                x4k,
                batch_size=train_config.batch_size,
                shuffle=True,
                drop_last=True,
            )

        x4k_loader = DataLoader(
            x4k,
            batch_sampler=x4k_sampler,
            num_workers=max(1, data_config.num_workers // 2),
            pin_memory=data_config.pin_memory,
            collate_fn=x4k_collate,
        )

        if is_main_process():
            print(f"    X4K: {len(x4k)} samples (steps/N={list(zip(x4k_steps, x4k_n_frames))})")
        return x4k_loader, x4k_sampler

    if mode == "mixed":
        # Mixed training: Vimeo (N=2) + X4K (variable N per step)
        if is_main_process():
            print("  Using mixed Vimeo + X4K training (variable N)...")

        # Create Vimeo dataset (N=2 frames)
        vimeo = VimeoTriplet(
            root=data_config.vimeo_root,
            split="train",
            mode=data_config.vimeo_mode,
            crop_size=data_config.crop_size,
            aug_flip=True,
        )

        # Resolve X4K root (handles path/train -> path)
        x4k_root = resolve_x4k_root(data_config.x4k_root)

        # Create X4K dataset with variable N per step (like TEMPO)
        x4k = X4K1000Dataset(
            root=x4k_root,
            split="train",
            steps=x4k_steps,      # e.g., [5, 31, 31]
            n_frames=x4k_n_frames, # e.g., [4, 3, 2]
            crop_size=data_config.x4k_crop_size,
            aug_flip=True,
        )

        if is_main_process():
            print(f"    Vimeo: {len(vimeo)} samples (N=2)")
            step_n_pairs = list(zip(x4k_steps, x4k_n_frames))
            print(f"    X4K: {len(x4k)} samples (steps/N={step_n_pairs})")

        # Create separate loaders for Vimeo and X4K
        # Vimeo loader
        if world_size > 1:
            vimeo_sampler = DistributedSampler(vimeo, shuffle=True)
        else:
            vimeo_sampler = None

        vimeo_loader = DataLoader(
            vimeo,
            batch_size=train_config.batch_size,
            shuffle=(vimeo_sampler is None),
            sampler=vimeo_sampler,
            num_workers=data_config.num_workers // 2,  # Split workers
            pin_memory=data_config.pin_memory,
            drop_last=True,
            collate_fn=vimeo_collate,
        )

        # X4K loader with batch sampler that groups by N
        if world_size > 1:
            x4k_sampler = DistributedX4KBatchSampler(
                x4k,
                batch_size=train_config.batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            x4k_sampler = X4KBatchSampler(
                x4k,
                batch_size=train_config.batch_size,
                shuffle=True,
                drop_last=True,
            )

        x4k_loader = DataLoader(
            x4k,
            batch_sampler=x4k_sampler,
            num_workers=max(1, data_config.num_workers // 2),
            pin_memory=data_config.pin_memory,
            collate_fn=x4k_collate,
        )

        if is_main_process():
            print(f"    Vimeo batches: {len(vimeo_loader)}, X4K batches: {len(x4k_sampler)}")
            print(f"    Dataset ratios: {dataset_ratios}")

        # Return MixedDataLoaders wrapper
        mixed_loader = MixedDataLoaders(
            vimeo_loader, x4k_loader,
            ratios=dataset_ratios,
        )

        # Return both samplers for epoch updates
        return mixed_loader, (vimeo_sampler, x4k_sampler)

    else:
        # Vimeo-only training
        dataset = VimeoTriplet(
            root=data_config.vimeo_root,
            split="train",
            mode=data_config.vimeo_mode,
            crop_size=data_config.crop_size,
            aug_flip=True,
        )

        if world_size > 1:
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = None

        loader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            drop_last=True,
            collate_fn=vimeo_collate,
        )

        if is_main_process():
            print(f"  Train (Vimeo only): {len(dataset)} samples")

        return loader, sampler


def sample_coordinates(target_frame: torch.Tensor,
                       num_samples: int,
                       target_time: torch.Tensor) -> tuple:
    """Sample random coordinates and corresponding RGB values."""
    B, C, H, W = target_frame.shape
    device = target_frame.device

    x = torch.rand(B, num_samples, device=device) * 2 - 1
    y = torch.rand(B, num_samples, device=device) * 2 - 1
    t = target_time.unsqueeze(1).expand(-1, num_samples)
    coords = torch.stack([x, y, t], dim=-1)

    grid = torch.stack([x, y], dim=-1).view(B, num_samples, 1, 2)
    target_rgb = F.grid_sample(
        target_frame, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(-1).permute(0, 2, 1)

    return coords, target_rgb


def sample_rgb_from_frame(frame: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample RGB values from a full frame at given coordinates."""
    grid = coords[..., :2].unsqueeze(2)  # [B, Q, 1, 2]
    rgb = F.grid_sample(
        frame, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(-1).permute(0, 2, 1)  # [B, Q, 3]
    return rgb


def train_epoch(model, dataloader, optimizer, scaler, criterion, config, epoch, writer, rank):
    """Train for one epoch."""
    model.train()

    # Update progressive curriculum if using advanced loss
    if hasattr(criterion, 'update_for_epoch'):
        active_losses = criterion.update_for_epoch(epoch)
        if is_main_process():
            print(f"  Active losses: {active_losses}")

    total_losses = defaultdict(float)
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    dataset_totals = defaultdict(lambda: {'loss': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'samples': 0})

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())

    training_mode = getattr(config.train, 'training_mode', 'hybrid')
    use_inr = training_mode in ("coordinate", "hybrid")
    use_full = training_mode in ("full_image", "hybrid")
    distill_enabled = getattr(config.train, 'distill_enabled', False)
    distill_weight = float(getattr(config.train, 'distill_weight', 0.0))
    distill_start_epoch = int(getattr(config.train, 'distill_start_epoch', 0))

    # Check if using HybridLoss (for hybrid decoder)
    is_hybrid_loss = isinstance(criterion, HybridLoss)
    net = model.module if hasattr(model, 'module') else model
    is_hybrid_decoder = getattr(net, 'use_hybrid_decoder', False)

    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 5:
            input_frames, input_times, target_time, target_frame, dataset_name = batch
        else:
            input_frames, input_times, target_time, target_frame = batch
            dataset_name = "vimeo"

        input_frames = input_frames.to(rank)
        input_times = input_times.to(rank)
        target_frame = target_frame.to(rank)

        if isinstance(target_time, (int, float)):
            target_time = torch.full((input_frames.shape[0],), target_time, device=rank)
        else:
            target_time = target_time.to(rank)

        optimizer.zero_grad()

        amp_dtype = torch.bfloat16 if config.train.amp_dtype == "bfloat16" else torch.float16
        with amp.autocast(device_type='cuda', enabled=config.train.use_amp, dtype=amp_dtype):
            total_loss = torch.tensor(0.0, device=rank)
            losses = {}
            pred_full = None
            pred_rgb = None
            target_rgb = None
            coords = None
            full_mode = None

            # Determine rendering mode
            if is_hybrid_decoder:
                full_mode = 'hybrid'
            elif getattr(net, 'use_nafnet_decoder', False):
                full_mode = 'nafnet'
            elif use_full:
                full_mode = 'fast'

            # Hybrid Loss: combined full-frame + coordinate training
            if is_hybrid_loss and is_hybrid_decoder:
                # Sample coordinates for SIREN guidance
                coords, target_rgb = sample_coordinates(
                    target_frame, config.train.num_samples, target_time
                )

                # Forward: get both coordinate predictions and full-frame
                pred_rgb, pred_full = model(
                    input_frames, input_times, coords,
                    target_time=target_time,
                    return_full=True,
                    full_mode=full_mode,
                )

                # Call HybridLoss with both predictions
                losses = criterion(
                    pred_full=pred_full,
                    target_full=target_frame,
                    pred_coords=pred_rgb,
                    target_coords=target_rgb,
                    model=net,
                    input_frames=input_frames,
                    input_times=input_times,
                )
                total_loss = losses['total']

            # Standard training: coordinate-based or full-frame
            else:
                if use_full:
                    full_mode = 'nafnet' if getattr(net, 'use_nafnet_decoder', False) else 'fast'

                if use_inr:
                    coords, target_rgb = sample_coordinates(
                        target_frame, config.train.num_samples, target_time
                    )
                    if use_full:
                        pred_rgb, pred_full = model(
                            input_frames, input_times, coords,
                            target_time=target_time,
                            return_full=True,
                            full_mode=full_mode,
                        )
                    else:
                        pred_rgb = model(input_frames, input_times, coords)

                    # Compute coordinate loss
                    if hasattr(criterion, 'uncertainty'):
                        coord_losses = criterion(
                            pred_rgb, target_rgb,
                            model=model,
                            input_frames=input_frames,
                            input_times=input_times,
                            compute_all=False
                        )
                    else:
                        coord_losses = criterion(pred_rgb, target_rgb, compute_all=False)

                    for name, val in coord_losses.items():
                        losses[f"coord_{name}"] = val
                    total_loss = total_loss + config.train.coordinate_loss_weight * coord_losses['total']

                if use_full and pred_full is None:
                    pred_full = model(
                        input_frames, input_times, None,
                        target_time=target_time,
                        return_full=True,
                        full_mode=full_mode,
                    )
                    full_losses = criterion(pred_full, target_frame, compute_all=True)
                    for name, val in full_losses.items():
                        losses[f"full_{name}"] = val
                    total_loss = total_loss + config.train.full_image_loss_weight * full_losses['total']

                if (distill_enabled and distill_weight > 0 and use_inr and use_full and
                        pred_full is not None and pred_rgb is not None and epoch >= distill_start_epoch):
                    naf_rgb = sample_rgb_from_frame(pred_full, coords)
                    diff = naf_rgb - pred_rgb
                    distill = torch.sqrt(diff * diff + config.train.charbonnier_eps ** 2).mean()
                    losses['distill'] = distill
                    total_loss = total_loss + distill_weight * distill

                losses['total'] = total_loss

        full_ssim = None
        if use_full and pred_full is not None:
            mse = F.mse_loss(pred_full.float(), target_frame.float())
            psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
            full_ssim = _compute_ssim(pred_full.detach(), target_frame.detach())
        else:
            mse = F.mse_loss(pred_rgb.float(), target_rgb.float())
            psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))

        scaler.scale(losses['total']).backward()

        if config.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        for name, val in losses.items():
            if isinstance(val, torch.Tensor):
                total_losses[name] += val.item() * input_frames.shape[0]
        total_psnr += psnr.item() * input_frames.shape[0]
        if full_ssim is not None:
            total_ssim += full_ssim * input_frames.shape[0]
        total_samples += input_frames.shape[0]

        ds = dataset_totals[dataset_name]
        ds['loss'] += losses['total'].item() * input_frames.shape[0]
        ds['psnr'] += psnr.item() * input_frames.shape[0]
        if full_ssim is not None:
            ds['ssim'] += full_ssim * input_frames.shape[0]
        ds['samples'] += input_frames.shape[0]

        if is_main_process():
            postfix = {
                'ds': dataset_name,
                'loss': f"{losses['total'].item():.4f}",
                'psnr': f"{psnr.item():.2f}",
            }
            if full_ssim is not None:
                postfix['ssim'] = f"{full_ssim:.4f}"
            if 'coord_charbonnier' in losses:
                postfix['charb'] = f"{losses['coord_charbonnier'].item():.4f}"
            pbar.set_postfix(postfix)

        # Tensorboard logging
        global_step = epoch * len(dataloader) + batch_idx
        if writer and batch_idx % config.train.log_every == 0:
            for name, val in losses.items():
                if isinstance(val, torch.Tensor):
                    writer.add_scalar(f'train/{name}', val.item(), global_step)
            writer.add_scalar('train/psnr', psnr.item(), global_step)
            if full_ssim is not None:
                writer.add_scalar('train/ssim', full_ssim, global_step)
            writer.add_scalar(f"train/{dataset_name}_psnr", psnr.item(), global_step)
            writer.add_scalar(f"train/{dataset_name}_loss", losses['total'].item(), global_step)
            if full_ssim is not None:
                writer.add_scalar(f"train/{dataset_name}_ssim", full_ssim, global_step)

    metrics = {name: val / max(total_samples, 1) for name, val in total_losses.items()}
    metrics['loss'] = metrics.get('total', 0.0)
    metrics['psnr'] = total_psnr / max(total_samples, 1)
    metrics['ssim'] = total_ssim / max(total_samples, 1) if total_samples > 0 else 0.0
    metrics['datasets'] = {}
    for name, ds in dataset_totals.items():
        if ds['samples'] == 0:
            continue
        metrics['datasets'][name] = {
            'loss': ds['loss'] / ds['samples'],
            'psnr': ds['psnr'] / ds['samples'],
            'ssim': ds['ssim'] / ds['samples'] if ds['samples'] > 0 else 0.0,
        }
    return metrics


@torch.no_grad()
def validate_dataset(model, dataloader, criterion, dataset_name: str, rank: int, world_size: int,
                     val_samples_dir: Path = None, save_samples: bool = False):
    """Validate on a single dataset."""
    model.eval()

    total_psnr = 0
    total_ssim = 0
    total_loss = 0
    num_samples = 0
    samples_saved = False

    for input_frames, input_times, target_time, target_frame in tqdm(
        dataloader, desc=f"Val {dataset_name}", disable=not is_main_process()
    ):
        input_frames = input_frames.to(rank)
        input_times = input_times.to(rank)
        target_frame = target_frame.to(rank)

        if isinstance(target_time, (int, float)):
            target_time_val = target_time
        else:
            target_time_val = target_time[0].item()

        B = target_frame.shape[0]

        # Render full frame
        net = model.module if hasattr(model, 'module') else model
        if getattr(net, 'use_hybrid_decoder', False):
            full_mode = 'hybrid'
        elif getattr(net, 'use_nafnet_decoder', False):
            full_mode = 'nafnet'
        else:
            full_mode = 'fast'
        pred_frame = net.render_frame(
            input_frames, input_times,
            target_time=target_time_val,
            mode=full_mode
        )

        # Compute losses
        losses = criterion(pred_frame, target_frame, compute_all=True)
        total_loss += losses['total'].item() * B

        # PSNR
        mse = F.mse_loss(pred_frame, target_frame)
        psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))

        # SSIM (standard gaussian-window SSIM)
        ssim = _compute_ssim(pred_frame, target_frame)

        total_psnr += psnr.item() * B
        total_ssim += ssim * B
        num_samples += B

        # Save a small set of validation samples (rank 0 only)
        if save_samples and not samples_saved and is_main_process() and val_samples_dir is not None:
            out_dir = val_samples_dir / dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            # Save first sample: input frames, prediction, target
            inp0 = input_frames[0, 0:1].clamp(0, 1)
            inp1 = input_frames[0, 1:2].clamp(0, 1)
            pred = pred_frame[0:1].clamp(0, 1)
            tgt = target_frame[0:1].clamp(0, 1)
            grid = torch.cat([inp0, inp1, pred, tgt], dim=0)
            save_image(grid, out_dir / "sample_0.png", nrow=4)
            samples_saved = True

    # Gather metrics across all GPUs
    if world_size > 1:
        metrics = torch.tensor([total_psnr, total_ssim, total_loss, num_samples], device=rank)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_psnr, total_ssim, total_loss, num_samples = metrics.tolist()

    return {
        'psnr': total_psnr / num_samples,
        'ssim': total_ssim / num_samples,
        'loss': total_loss / num_samples,
    }


class InputPadder:
    """Pads images to be divisible by divisor."""

    def __init__(self, dims, divisor=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (divisor - (self.ht % divisor)) % divisor
        pad_wd = (divisor - (self.wd % divisor)) % divisor
        self._pad = [0, pad_wd, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        return x[..., :self.ht, :self.wd]


@torch.no_grad()
def validate_x4k_cascaded(model, dataloader, rank: int, world_size: int, scale: str = "4k",
                          val_samples_dir: Path = None, save_samples: bool = False):
    """
    Validate on X4K using cascaded 8x interpolation.

    Cascaded prediction approach:
    - Level 0: N=2 (frames 0, 32) → predict frame 16
    - Level 1: N=3 (frames 0, 16, 32) → predict frames 8, 24
    - Level 2: N=4 (closest anchors) → predict frames 4, 12, 20, 28

    Args:
        model: SPACE model
        dataloader: DataLoader for X4KSequenceDataset
        rank: GPU rank
        world_size: Number of GPUs
        scale: "4k" or "2k"

    Returns:
        dict with PSNR and SSIM metrics
    """
    model.eval()
    device = rank if isinstance(rank, torch.device) else torch.device(f'cuda:{rank}')

    # Get the actual model (unwrap DDP if needed)
    net = model.module if hasattr(model, 'module') else model

    target_indices = [4, 8, 12, 16, 20, 24, 28]  # 8x interpolation

    all_psnr = []
    all_ssim = []

    samples_saved = False

    for frame_0, frame_32, gt_frames_list, metadata in tqdm(
        dataloader, desc=f"Val X4K-{scale} (cascaded)", disable=not is_main_process()
    ):
        frame_0 = frame_0.to(device)
        frame_32 = frame_32.to(device)

        B = frame_0.shape[0]

        # Process each sample in batch
        for b in range(B):
            f0 = frame_0[b:b+1]  # [1, 3, H, W]
            f32 = frame_32[b:b+1]
            gt_frames = gt_frames_list[b]  # dict {idx: tensor}

            # Pad for model
            padder = InputPadder(f0.shape, divisor=32)
            f0_pad, f32_pad = padder.pad(f0, f32)

            # Initialize anchor frames dict with padded inputs
            anchor_frames = {0: f0_pad, 32: f32_pad}
            predictions = {}

            # Level 0: N=2 (0, 32) → predict 16
            pred_16 = _predict_with_anchors(
                net, anchor_frames, [0, 32], target_idx=16, device=device
            )
            anchor_frames[16] = pred_16
            predictions[16] = padder.unpad(pred_16)

            # Level 1: N=3 (0, 16, 32) → predict 8, 24
            for target_idx in [8, 24]:
                pred = _predict_with_anchors(
                    net, anchor_frames, [0, 16, 32], target_idx=target_idx, device=device
                )
                anchor_frames[target_idx] = pred
                predictions[target_idx] = padder.unpad(pred)

            # Level 2: N=4 (closest anchors) → predict 4, 12, 20, 28
            # Available anchors: 0, 8, 16, 24, 32
            for target_idx in [4, 12, 20, 28]:
                # Find closest 4 anchors
                anchors = _get_closest_anchors(list(anchor_frames.keys()), target_idx, n=4)
                pred = _predict_with_anchors(
                    net, anchor_frames, anchors, target_idx=target_idx, device=device
                )
                predictions[target_idx] = padder.unpad(pred)

            # Compute metrics for all target frames
            seq_psnr = []
            seq_ssim = []

            for target_idx in target_indices:
                pred = predictions[target_idx]  # [1, 3, H, W]
                gt = gt_frames[target_idx].unsqueeze(0).to(device)  # [1, 3, H, W]

                # PSNR
                mse = F.mse_loss(pred, gt)
                psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
                seq_psnr.append(psnr.item())

                # SSIM (gaussian-window SSIM)
                ssim_val = _compute_ssim(pred, gt)
                seq_ssim.append(ssim_val)

            all_psnr.append(sum(seq_psnr) / len(seq_psnr))
            all_ssim.append(sum(seq_ssim) / len(seq_ssim))

            if save_samples and not samples_saved and is_main_process() and val_samples_dir is not None:
                out_dir = val_samples_dir / f"x4k_{scale}"
                out_dir.mkdir(parents=True, exist_ok=True)
                pred = predictions[16].clamp(0, 1)
                gt = gt_frames[16].unsqueeze(0).to(device).clamp(0, 1)
                grid = torch.cat([
                    padder.unpad(f0_pad).clamp(0, 1),
                    padder.unpad(f32_pad).clamp(0, 1),
                    pred,
                    gt,
                ], dim=0)
                save_image(grid, out_dir / "sample_0.png", nrow=4)
                samples_saved = True

    # Gather metrics across GPUs
    if world_size > 1:
        # Gather all values
        psnr_tensor = torch.tensor(all_psnr, device=device)
        ssim_tensor = torch.tensor(all_ssim, device=device)

        psnr_list = [torch.zeros_like(psnr_tensor) for _ in range(world_size)]
        ssim_list = [torch.zeros_like(ssim_tensor) for _ in range(world_size)]

        dist.all_gather(psnr_list, psnr_tensor)
        dist.all_gather(ssim_list, ssim_tensor)

        all_psnr = torch.cat(psnr_list).tolist()
        all_ssim = torch.cat(ssim_list).tolist()

    avg_psnr = sum(all_psnr) / len(all_psnr) if all_psnr else 0
    avg_ssim = sum(all_ssim) / len(all_ssim) if all_ssim else 0

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'loss': 0.0,  # Not computed for cascaded eval
    }


def _predict_with_anchors(model, anchor_frames, anchor_indices, target_idx, device):
    """Predict a frame using specified anchors."""
    N = len(anchor_indices)

    # Build input tensors
    frames_list = [anchor_frames[i] for i in anchor_indices]
    frames = torch.cat(frames_list, dim=0).unsqueeze(0)  # [1, N, 3, H, W]

    # Compute normalized times
    first_idx, last_idx = anchor_indices[0], anchor_indices[-1]
    anchor_times = torch.tensor(
        [(i - first_idx) / (last_idx - first_idx) for i in anchor_indices],
        dtype=torch.float32, device=device
    ).unsqueeze(0)  # [1, N]

    target_time = (target_idx - first_idx) / (last_idx - first_idx)

    # Predict - use hybrid mode if available
    if getattr(model, 'use_hybrid_decoder', False):
        full_mode = 'hybrid'
    elif getattr(model, 'use_nafnet_decoder', False):
        full_mode = 'nafnet'
    else:
        full_mode = 'fast'
    pred = model.render_frame(
        frames, anchor_times,
        target_time=target_time,
        mode=full_mode
    )

    return pred


def _get_closest_anchors(available, target_idx, n=4):
    """Get n closest anchors to target that bracket it."""
    sorted_anchors = sorted(available)

    before = [i for i in sorted_anchors if i < target_idx]
    after = [i for i in sorted_anchors if i > target_idx]

    selected = []

    # Add immediate neighbors
    if before:
        selected.append(before[-1])
    if after:
        selected.append(after[0])

    # Add more anchors if available
    remaining_before = before[:-1][::-1] if before else []
    remaining_after = after[1:] if after else []

    while len(selected) < n and (remaining_before or remaining_after):
        if remaining_before:
            selected.append(remaining_before.pop(0))
        if len(selected) < n and remaining_after:
            selected.append(remaining_after.pop(0))

    return sorted(selected)


def _compute_ssim(pred, target, window_size: int = 11, sigma: float = 1.5):
    """Compute SSIM using a Gaussian window (expects inputs in [0, 1])."""
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)

    device = pred.device
    dtype = pred.dtype
    channels = pred.shape[1]

    # Create Gaussian window
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)
    window = (g.transpose(2, 1) @ g).view(1, 1, window_size, window_size)
    window = window.expand(channels, 1, window_size, window_size)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def validate_all(model, val_loaders, criterion, epoch, writer, rank, world_size,
                 val_samples_dir: Path = None, save_samples: bool = False):
    """
    Validate on all datasets and compute average metrics.

    Uses cascaded 8x interpolation for X4K validation (x4k_2k, x4k_4k).
    Uses standard validation for Vimeo.

    Returns:
        dict with per-dataset and average metrics
    """
    all_metrics = {}
    psnr_values = []
    ssim_values = []

    for name, loader in val_loaders.items():
        # Use cascaded validation for X4K
        if name.startswith('x4k_'):
            scale = name.split('_')[1]  # '2k' or '4k'
            metrics = validate_x4k_cascaded(
                model, loader, rank, world_size, scale=scale,
                val_samples_dir=val_samples_dir, save_samples=save_samples
            )
        else:
            # Standard validation for Vimeo
            metrics = validate_dataset(
                model, loader, criterion, name, rank, world_size,
                val_samples_dir=val_samples_dir, save_samples=save_samples
            )

        all_metrics[name] = metrics
        psnr_values.append(metrics['psnr'])
        ssim_values.append(metrics['ssim'])

        if is_main_process() and writer:
            writer.add_scalar(f'val/{name}_psnr', metrics['psnr'], epoch)
            writer.add_scalar(f'val/{name}_ssim', metrics['ssim'], epoch)
            if 'loss' in metrics:
                writer.add_scalar(f'val/{name}_loss', metrics['loss'], epoch)

    # Compute average across datasets
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0

    all_metrics['avg'] = {'psnr': avg_psnr, 'ssim': avg_ssim}

    if is_main_process() and writer:
        writer.add_scalar('val/avg_psnr', avg_psnr, epoch)
        writer.add_scalar('val/avg_ssim', avg_ssim, epoch)

    return all_metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, config, path, criterion=None):
    """Save training checkpoint."""
    # Get model state dict (handle DDP wrapper)
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'config': {
            'model': config.model.__dict__,
            'data': config.data.__dict__,
            'train': config.train.__dict__,
        },
    }
    if criterion is not None and hasattr(criterion, 'state_dict'):
        checkpoint['criterion_state_dict'] = criterion.state_dict()
    torch.save(checkpoint, path)


def main_worker(rank: int, world_size: int, args):
    """Main training worker for each GPU."""

    # Silence torchvision pretrained/weights deprecation warnings from third-party libs (e.g. LPIPS).
    warnings.filterwarnings(
        "ignore",
        message="The parameter 'pretrained' is deprecated since 0.13",
        category=UserWarning,
        module=r"torchvision\.models\._utils",
    )
    warnings.filterwarnings(
        "ignore",
        message="Arguments other than a weight enum or `None` for 'weights' are deprecated",
        category=UserWarning,
        module=r"torchvision\.models\._utils",
    )

    # Setup DDP
    if world_size > 1:
        setup_ddp(rank, world_size)
    os.environ["SPACE_RANK"] = str(rank)

    # Load config
    if args.config:
        config = Config.load(args.config)
    elif args.fast:
        config = get_fast_config()
    elif getattr(args, 'mixed', False):
        config = get_mixed_config()
    elif getattr(args, 'no_uncertainty', False):
        config = get_no_uncertainty_config()
    else:
        config = get_default_config()

    # Override with CLI arguments
    if args.vimeo_root:
        config.data.vimeo_root = args.vimeo_root
    if getattr(args, 'x4k_root', None):
        config.data.x4k_root = args.x4k_root
        config.data.use_mixed_training = True  # Enable mixed training if X4K root provided
    if args.vimeo_val:
        config.data.vimeo_val_root = args.vimeo_val
    if args.x4k_val:
        config.data.x4k_val_root = args.x4k_val
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.epochs:
        config.train.epochs = args.epochs
    # Enable mixed training if explicitly requested
    if getattr(args, 'mixed', False):
        config.data.use_mixed_training = True
    # X4K TEMPO-style configuration
    if getattr(args, 'x4k_steps', None):
        config.data.x4k_steps = args.x4k_steps
    if getattr(args, 'x4k_n_frames', None):
        config.data.x4k_n_frames = args.x4k_n_frames
        # Validate lengths match
        if len(config.data.x4k_n_frames) != len(config.data.x4k_steps):
            raise ValueError(
                f"x4k_n_frames ({len(config.data.x4k_n_frames)}) must match "
                f"x4k_steps ({len(config.data.x4k_steps)})"
            )

    # Optimization options
    if getattr(args, 'compile', False):
        config.train.use_compile = True
    if getattr(args, 'compile_mode', None):
        config.train.compile_mode = args.compile_mode
    if getattr(args, 'gradient_checkpointing', False):
        config.train.use_gradient_checkpointing = True

    # Model options
    if getattr(args, 'refinement', False):
        config.model.use_refinement = True

    set_seed(config.train.seed, rank)

    # Create output directories (only on main process)
    if is_main_process():
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_root = Path("runs") / timestamp
        checkpoint_dir = run_root / "checkpoints"
        logs_dir = run_root / "logs"
        val_samples_dir = run_root / "validation samples"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        val_samples_dir.mkdir(parents=True, exist_ok=True)
        config.save(run_root / "config.json")
        writer = SummaryWriter(logs_dir)
        print(f"\n{'='*60}")
        print(f"SPACE DDP Training - {world_size} GPU(s)")
        print(f"{'='*60}")
    else:
        checkpoint_dir = None
        val_samples_dir = None
        writer = None

    # Create model
    if is_main_process():
        print("\nCreating model...")
    model = create_model(config)

    # Enable gradient checkpointing (before moving to GPU for memory savings)
    if config.train.use_gradient_checkpointing:
        model.set_gradient_checkpointing(True)
        if is_main_process():
            print("  Gradient checkpointing: enabled")

    model = model.to(rank)

    # Apply torch.compile (before DDP for best results)
    if config.train.use_compile:
        if is_main_process():
            print(f"  Compiling model with mode='{config.train.compile_mode}'...")
        model = torch.compile(model, mode=config.train.compile_mode)
        if is_main_process():
            print("  Model compiled successfully")

    if world_size > 1:
        training_mode = getattr(config.train, 'training_mode', 'hybrid')
        use_inr = training_mode in ("coordinate", "hybrid")
        use_full = training_mode in ("full_image", "hybrid")
        uses_hybrid = getattr(config.model, 'use_hybrid_decoder', True)
        uses_nafnet = use_full and config.model.use_nafnet_decoder and not uses_hybrid
        uses_fast = use_full and (not config.model.use_nafnet_decoder) and config.model.use_fast_decoder and not uses_hybrid
        # Any optional decoder that isn't exercised this run will be unused.
        # With hybrid decoder, both NAFNet and SIREN guidance are used
        find_unused = (
            (not use_inr and not uses_hybrid) or
            (config.model.use_fast_decoder and not uses_fast) or
            (config.model.use_nafnet_decoder and not uses_nafnet and not uses_hybrid)
        )
        model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused)

    if is_main_process():
        base_model = model.module if hasattr(model, 'module') else model
        # Handle compiled model wrapper
        if hasattr(base_model, '_orig_mod'):
            base_model = base_model._orig_mod
        param_counts = base_model.get_param_count()
        print(f"Model parameters: {param_counts['total']:,}")

    # Create loss function
    if is_main_process():
        print("\nCreating loss function...")

    # Check if using hybrid decoder - use HybridLoss optimized for full-frame training
    use_hybrid = getattr(config.model, 'use_hybrid_decoder', True)
    use_hybrid_loss = getattr(config.train, 'use_hybrid_loss', use_hybrid)

    if use_hybrid_loss and use_hybrid:
        if is_main_process():
            uw_status = "enabled" if config.train.use_uncertainty_weighting else "disabled"
            print(f"  Using HybridLoss for NAFNet+SIREN decoder (uncertainty: {uw_status})")
        criterion = HybridLoss(
            use_charbonnier=True,
            use_msssim=config.train.use_msssim,
            use_lpips=config.train.use_lpips,
            use_gradient=config.train.use_gradient,
            use_frequency=config.train.use_frequency_adv,
            use_anchor_distillation=config.train.use_anchor_distillation,
            anchor_distill_weight=getattr(config.train, 'hybrid_anchor_weight', 0.3),
            anchor_indices='first_last',
            use_coord_loss=True,
            coord_samples=getattr(config.train, 'hybrid_coord_samples', 1024),
            use_uncertainty_weighting=config.train.use_uncertainty_weighting,
            use_progressive=config.train.use_progressive_loss,
            lpips_net=config.train.lpips_net,
        )
    elif config.train.use_advanced_loss:
        if is_main_process():
            uw_status = "enabled" if config.train.use_uncertainty_weighting else "disabled"
            print(f"  Using AdvancedSPACELoss (uncertainty: {uw_status})")
        criterion = AdvancedSPACELoss(
            use_uncertainty_weighting=config.train.use_uncertainty_weighting,
            use_msssim=config.train.use_msssim,
            use_lpips=config.train.use_lpips,
            use_gradient=config.train.use_gradient,
            use_laplacian=config.train.use_laplacian,
            use_frequency=config.train.use_frequency_adv,
            use_anchor_distillation=config.train.use_anchor_distillation,
            anchor_distill_weight=config.train.anchor_distill_weight,
            anchor_distill_indices=config.train.anchor_distill_indices,
            anchor_distill_samples=config.train.anchor_distill_samples,
            use_progressive=config.train.use_progressive_loss,
            lpips_net=config.train.lpips_net,
        )
    else:
        if is_main_process():
            print("  Using SPACELoss")
        criterion = SPACELoss(
            charbonnier_weight=config.train.charbonnier_weight,
            charbonnier_eps=config.train.charbonnier_eps,
            ssim_weight=config.train.ssim_weight,
            perceptual_weight=config.train.perceptual_weight,
            frequency_weight=config.train.frequency_weight,
            use_perceptual=config.train.use_perceptual,
        )
    criterion = criterion.to(rank)

    # Create dataloaders
    if is_main_process():
        print("\nCreating dataloaders...")
    val_loaders = create_validation_loaders(config, rank, world_size)

    train_loader = None
    train_sampler = None
    curriculum_key = None

    def setup_train_loader_for_epoch(ep: int):
        nonlocal train_loader, train_sampler, curriculum_key
        settings = get_curriculum_settings(config, ep)
        key = (
            settings['mode'],
            tuple(settings['x4k_steps']) if settings['x4k_steps'] is not None else None,
            tuple(settings['x4k_n_frames']) if settings['x4k_n_frames'] is not None else None,
            tuple(settings['dataset_ratios']) if settings['dataset_ratios'] is not None else None,
        )
        if key != curriculum_key:
            train_loader, train_sampler = create_train_loader(
                config, rank, world_size,
                mode=settings['mode'],
                x4k_steps=settings['x4k_steps'],
                x4k_n_frames=settings['x4k_n_frames'],
                dataset_ratios=settings['dataset_ratios'],
            )
            curriculum_key = key
            if is_main_process():
                print(f"  Curriculum phase: {settings['name']}")
        return train_loader, train_sampler

    if not val_loaders:
        if is_main_process():
            print("  WARNING: No validation datasets configured!")

    # Create optimizer
    # Check if criterion has uncertainty weighting (both AdvancedSPACELoss and HybridLoss may have it)
    has_uncertainty = hasattr(criterion, 'uncertainty') and config.train.use_uncertainty_weighting
    if has_uncertainty:
        optimizer = torch.optim.AdamW([
            {'params': (model.module if hasattr(model, 'module') else model).parameters(),
             'lr': config.train.learning_rate},
            {'params': criterion.uncertainty.parameters(),
             'lr': config.train.learning_rate * 0.1},
        ], weight_decay=config.train.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

    # Create scheduler
    if config.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs,
            eta_min=config.train.min_lr,
        )
    else:
        scheduler = None

    scaler = amp.GradScaler('cuda', enabled=config.train.use_amp)

    # Resume from checkpoint
    start_epoch = 0
    best_avg_psnr = 0
    if args.resume:
        if is_main_process():
            print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}')

        base_model = model.module if hasattr(model, 'module') else model
        base_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'criterion_state_dict' in checkpoint and hasattr(criterion, 'load_state_dict'):
            criterion.load_state_dict(checkpoint['criterion_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_avg_psnr = checkpoint['metrics'].get('avg', {}).get('psnr', 0)

    # Training loop
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Starting training from epoch {start_epoch}")
        print(f"{'='*60}\n")

    for epoch in range(start_epoch, config.train.epochs):
        # Rebuild train loader if curriculum phase changes
        train_loader, train_sampler = setup_train_loader_for_epoch(epoch)

        # Set epoch for distributed sampler(s)
        if train_sampler is not None:
            if isinstance(train_sampler, tuple):
                # Mixed training: (vimeo_sampler, x4k_sampler)
                vimeo_sampler, x4k_sampler = train_sampler
                if vimeo_sampler is not None and hasattr(vimeo_sampler, 'set_epoch'):
                    vimeo_sampler.set_epoch(epoch)
                if x4k_sampler is not None and hasattr(x4k_sampler, 'set_epoch'):
                    x4k_sampler.set_epoch(epoch)
            elif hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, criterion,
            config, epoch, writer, rank
        )

        # Update scheduler
        if scheduler:
            scheduler.step()
            if writer:
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
        if writer and 'datasets' in train_metrics:
            for name, m in train_metrics['datasets'].items():
                writer.add_scalar(f"train/{name}_epoch_loss", m['loss'], epoch)
                writer.add_scalar(f"train/{name}_epoch_psnr", m['psnr'], epoch)
                if 'ssim' in m:
                    writer.add_scalar(f"train/{name}_epoch_ssim", m['ssim'], epoch)

        # Validate on all datasets
        if epoch % config.train.val_every == 0 and val_loaders:
            save_samples = (epoch % 10 == 0)
            val_metrics = validate_all(
                model, val_loaders, criterion, epoch, writer, rank, world_size,
                val_samples_dir=val_samples_dir, save_samples=save_samples
            )

            if is_main_process():
                # Training per-dataset metrics
                train_ds_strs = []
                for name, m in train_metrics.get('datasets', {}).items():
                    train_ds_strs.append(f"train_{name}: PSNR={m['psnr']:.2f}")

                # Validation per-dataset metrics (PSNR + SSIM)
                metric_strs = []
                for name, m in val_metrics.items():
                    if name != 'avg':
                        metric_strs.append(f"{name}: PSNR={m['psnr']:.2f}/SSIM={m['ssim']:.4f}")

                print(f"Epoch {epoch}: Loss={train_metrics['loss']:.4f}, "
                      f"Train PSNR={train_metrics.get('psnr', 0):.2f}")
                if train_ds_strs:
                    print(f"  Train: {', '.join(train_ds_strs)}")
                print(f"  Val: {', '.join(metric_strs)}")
                print(f"  Avg: PSNR={val_metrics['avg']['psnr']:.2f}, SSIM={val_metrics['avg']['ssim']:.4f}")

                # Save best model based on average PSNR
                if val_metrics['avg']['psnr'] > best_avg_psnr:
                    best_avg_psnr = val_metrics['avg']['psnr']
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        epoch, val_metrics, config,
                        checkpoint_dir / "best.pt",
                        criterion=criterion
                    )
                    print(f"  -> New best! Avg PSNR: {best_avg_psnr:.2f}")
        else:
            if is_main_process():
                train_ds_strs = []
                for name, m in train_metrics.get('datasets', {}).items():
                    train_ds_strs.append(f"{name}: PSNR={m['psnr']:.2f}")

                print(f"Epoch {epoch}: Loss={train_metrics['loss']:.4f}, "
                      f"Train PSNR={train_metrics.get('psnr', 0):.2f}")
                if train_ds_strs:
                    print(f"  Train: {', '.join(train_ds_strs)}")

        # Save periodic checkpoint
        if is_main_process() and epoch % config.train.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, train_metrics, config,
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                criterion=criterion
            )

    # Save final model
    if is_main_process():
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            config.train.epochs - 1, train_metrics, config,
            checkpoint_dir / "final.pt",
            criterion=criterion
        )
        print(f"\n{'='*60}")
        print(f"Training complete! Best Avg PSNR: {best_avg_psnr:.2f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        print(f"{'='*60}")
        writer.close()

    if world_size > 1:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Train SPACE model with DDP")

    # Config
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    parser.add_argument('--fast', action='store_true', help="Use fast/tiny config")
    parser.add_argument('--mixed', action='store_true', help="Use mixed Vimeo+X4K training")
    parser.add_argument('--no-uncertainty', action='store_true', help="Disable uncertainty weighting")

    # Data paths
    parser.add_argument('--vimeo-root', type=str, default=None,
                        help="Vimeo triplet root (contains sequences/, tri_trainlist.txt, tri_testlist.txt)")
    parser.add_argument('--x4k-root', type=str, default=None,
                        help="X4K root (contains train/ and test/). Can also point to train/ directly.")
    parser.add_argument('--vimeo-val', type=str, default=None,
                        help="Override Vimeo validation path (default: same as vimeo-root)")
    parser.add_argument('--x4k-val', type=str, default=None,
                        help="Override X4K validation path (default: derived from x4k-root)")

    # X4K configuration (TEMPO-style)
    parser.add_argument('--x4k-steps', type=int, nargs='+', default=None,
                        help="X4K step values, e.g., --x4k-steps 5 31 31")
    parser.add_argument('--x4k-n-frames', type=int, nargs='+', default=None,
                        help="X4K n_frames per step, e.g., --x4k-n-frames 4 3 2")

    # Training
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint to resume from")

    # Optimization
    parser.add_argument('--compile', action='store_true', help="Use torch.compile()")
    parser.add_argument('--compile-mode', type=str, default=None,
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help="torch.compile mode (default: reduce-overhead)")
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help="Enable gradient checkpointing to reduce memory")

    # Coarse-to-Fine Refinement
    parser.add_argument('--refinement', action='store_true',
                        help="Enable coarse-to-fine refinement module")

    # DDP
    parser.add_argument('--gpus', type=int, default=None,
                        help="Number of GPUs (default: all available)")

    args = parser.parse_args()

    # Determine world size
    if args.gpus:
        world_size = args.gpus
    else:
        world_size = torch.cuda.device_count()

    if world_size == 0:
        print("No CUDA devices available!")
        sys.exit(1)

    print(f"Using {world_size} GPU(s)")

    if world_size == 1:
        # Single GPU - run directly
        main_worker(0, 1, args)
    else:
        # Multi-GPU - spawn processes
        import torch.multiprocessing as mp
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
