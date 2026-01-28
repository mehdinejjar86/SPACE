#!/usr/bin/env python3
# train.py
# Training script for SPACE

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import SPACE, build_space, SPACELoss, AdvancedSPACELoss
from data import get_dataloader, VimeoTriplet, X4K1000Dataset, PureBatchSampler, vimeo_collate
from config import Config, get_default_config, get_fast_config, get_mixed_config, get_no_uncertainty_config
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_mixed_dataloader(config, split: str = "train") -> DataLoader:
    """
    Create mixed dataloader for Vimeo + X4K training with pure batches.

    Uses PureBatchSampler to ensure each batch contains only one dataset type
    (prevents mixing N=2 Vimeo with N=4 X4K in same batch).
    """
    data_config = config.data
    train_config = config.train
    is_train = (split == "train")

    # Create Vimeo dataset (N=2 frames)
    vimeo = VimeoTriplet(
        root=data_config.vimeo_root,
        split=split,
        mode=data_config.vimeo_mode if is_train else "interp",
        crop_size=data_config.crop_size if is_train else None,
        aug_flip=is_train,
    )

    # Create X4K dataset (N=4 frames)
    x4k = X4K1000Dataset(
        root=data_config.x4k_root,
        split=split,
        steps=data_config.x4k_steps,
        crop_size=data_config.x4k_crop_size if is_train else None,
        aug_flip=is_train,
        n_frames=4,
    )

    # Concatenate datasets
    concat_dataset = ConcatDataset([vimeo, x4k])

    if is_train:
        # Use PureBatchSampler for training (no mixing within batches)
        sampler = PureBatchSampler(
            dataset_sizes=[len(vimeo), len(x4k)],
            batch_size=train_config.batch_size,
            ratios=data_config.dataset_ratios,
            shuffle=True,
            drop_last=True,
        )

        return DataLoader(
            concat_dataset,
            batch_sampler=sampler,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            collate_fn=mixed_collate_fn,
        )
    else:
        # For validation, just use Vimeo (simpler, consistent N)
        return DataLoader(
            vimeo,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            collate_fn=vimeo_collate,
        )


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


def create_model(config) -> SPACE:
    """Create SPACE model from config."""
    model_config = config.model

    # Use preset if specified
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
        )

    # Otherwise use explicit parameters
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
    )


def sample_coordinates(target_frame: torch.Tensor,
                       num_samples: int,
                       target_time: torch.Tensor) -> tuple:
    """
    Sample random coordinates and corresponding RGB values from target frame.

    Args:
        target_frame: [B, 3, H, W] target frame
        num_samples: Number of coordinates to sample
        target_time: [B] target timestamps

    Returns:
        coords: [B, Q, 3] sampled (x, y, t) coordinates
        target_rgb: [B, Q, 3] RGB values at those coordinates
    """
    B, C, H, W = target_frame.shape
    device = target_frame.device

    # Sample random normalized coordinates
    x = torch.rand(B, num_samples, device=device) * 2 - 1  # [-1, 1]
    y = torch.rand(B, num_samples, device=device) * 2 - 1  # [-1, 1]

    # Expand target_time to match samples
    t = target_time.unsqueeze(1).expand(-1, num_samples)

    # Build coordinate tensor
    coords = torch.stack([x, y, t], dim=-1)  # [B, Q, 3]

    # Sample RGB from target frame using grid_sample
    grid = torch.stack([x, y], dim=-1).view(B, num_samples, 1, 2)
    target_rgb = F.grid_sample(
        target_frame, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # [B, C, Q, 1]
    target_rgb = target_rgb.squeeze(-1).permute(0, 2, 1)  # [B, Q, C]

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


def _compute_ssim(pred, target, window_size: int = 11, sigma: float = 1.5):
    """Compute SSIM using a Gaussian window (expects inputs in [0, 1])."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    device = pred.device
    dtype = pred.dtype
    channels = pred.shape[1]

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


def train_epoch(model: SPACE,
                dataloader,
                optimizer,
                scaler: amp.GradScaler,
                criterion: SPACELoss,
                config,
                epoch: int,
                writer: SummaryWriter = None) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_charbonnier = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0

    training_mode = getattr(config.train, 'training_mode', 'coordinate')
    use_inr = training_mode in ("coordinate", "hybrid")
    use_full = training_mode in ("full_image", "hybrid")
    distill_enabled = getattr(config.train, 'distill_enabled', False)
    distill_weight = float(getattr(config.train, 'distill_weight', 0.0))
    distill_start_epoch = int(getattr(config.train, 'distill_start_epoch', 0))
    distill_enabled = getattr(config.train, 'distill_enabled', False)
    distill_weight = float(getattr(config.train, 'distill_weight', 0.0))
    distill_start_epoch = int(getattr(config.train, 'distill_start_epoch', 0))

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (input_frames, input_times, target_frame, target_time) in enumerate(pbar):
        device = next(model.parameters()).device
        input_frames = input_frames.to(device)
        input_times = input_times.to(device)
        target_frame = target_frame.to(device)

        # Handle target_time
        if isinstance(target_time, (int, float)):
            target_time = torch.full((input_frames.shape[0],), target_time, device=device)
        else:
            target_time = target_time.to(device)

        # Forward pass with mixed precision
        optimizer.zero_grad()

        amp_dtype = torch.bfloat16 if config.train.amp_dtype == "bfloat16" else torch.float16
        with amp.autocast(device_type='cuda', enabled=config.train.use_amp, dtype=amp_dtype):
            total = torch.tensor(0.0, device=device)
            losses = {}
            pred_full = None
            pred_rgb = None

            if use_inr:
                coords, target_rgb = sample_coordinates(
                    target_frame,
                    config.train.num_samples,
                    target_time
                )
                pred_rgb = model(input_frames, input_times, coords)
                coord_losses = criterion(pred_rgb, target_rgb, compute_all=False)
                for name, val in coord_losses.items():
                    losses[f"coord_{name}"] = val
                total = total + config.train.coordinate_loss_weight * coord_losses['total']

            if use_full:
                full_mode = 'nafnet' if getattr(model, 'use_nafnet_decoder', False) else 'fast'
                pred_full = model.render_frame(input_frames, input_times, target_time, mode=full_mode)
                full_losses = criterion(pred_full, target_frame, compute_all=True)
                for name, val in full_losses.items():
                    losses[f"full_{name}"] = val
                total = total + config.train.full_image_loss_weight * full_losses['total']

            if (distill_enabled and distill_weight > 0 and use_inr and use_full and
                    pred_full is not None and pred_rgb is not None and epoch >= distill_start_epoch):
                naf_rgb = sample_rgb_from_frame(pred_full, coords)
                diff = naf_rgb - pred_rgb
                distill = torch.sqrt(diff * diff + config.train.charbonnier_eps ** 2).mean()
                losses['distill'] = distill
                total = total + distill_weight * distill

            losses['total'] = total

        # Backward pass
        scaler.scale(losses['total']).backward()

        # Gradient clipping
        if config.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        total_loss += losses['total'].item()
        if 'coord_charbonnier' in losses:
            total_charbonnier += losses['coord_charbonnier'].item()
        full_ssim = None
        if use_full and pred_full is not None:
            mse = F.mse_loss(pred_full.float(), target_frame.float())
            full_ssim = _compute_ssim(pred_full.detach(), target_frame.detach())
        else:
            mse = F.mse_loss(pred_rgb.float(), target_rgb.float())
        total_psnr += (10 * torch.log10(1.0 / mse.clamp(min=1e-10))).item()
        if full_ssim is not None:
            total_ssim += full_ssim
        num_batches += 1

        # Update progress bar
        postfix = {'loss': f"{losses['total'].item():.4f}"}
        if 'coord_charbonnier' in losses:
            postfix['charb'] = f"{losses['coord_charbonnier'].item():.4f}"
        if full_ssim is not None:
            postfix['ssim'] = f"{full_ssim:.4f}"
        pbar.set_postfix(postfix)

        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        if writer and batch_idx % config.train.log_every == 0:
            for name, val in losses.items():
                if isinstance(val, torch.Tensor):
                    writer.add_scalar(f"train/{name}", val.item(), global_step)
            if full_ssim is not None:
                writer.add_scalar("train/ssim", full_ssim, global_step)

    return {
        'loss': total_loss / num_batches,
        'charbonnier': total_charbonnier / max(num_batches, 1),
        'psnr': total_psnr / max(num_batches, 1),
        'ssim': total_ssim / max(num_batches, 1),
    }


def train_epoch_advanced(model: SPACE,
                         dataloader,
                         optimizer,
                         scaler: amp.GradScaler,
                         criterion: AdvancedSPACELoss,
                         config,
                         epoch: int,
                         writer: SummaryWriter = None) -> dict:
    """
    Train for one epoch with advanced loss system.

    Features:
    - Anchor distillation for self-consistency
    - Uncertainty-weighted losses
    - Progressive curriculum
    - Periodic full-frame evaluation
    """
    model.train()

    # Update active losses based on epoch (progressive curriculum)
    active_losses = criterion.update_for_epoch(epoch)
    print(f"  Active losses: {active_losses}")

    total_losses = defaultdict(float)
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    training_mode = getattr(config.train, 'training_mode', 'coordinate')
    use_inr = training_mode in ("coordinate", "hybrid")
    use_full = training_mode in ("full_image", "hybrid")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (input_frames, input_times, target_frame, target_time) in enumerate(pbar):
        device = next(model.parameters()).device
        input_frames = input_frames.to(device)
        input_times = input_times.to(device)
        target_frame = target_frame.to(device)

        # Handle target_time
        if isinstance(target_time, (int, float)):
            target_time = torch.full((input_frames.shape[0],), target_time, device=device)
        else:
            target_time = target_time.to(device)

        optimizer.zero_grad()

        amp_dtype = torch.bfloat16 if config.train.amp_dtype == "bfloat16" else torch.float16
        with amp.autocast(device_type='cuda', enabled=config.train.use_amp, dtype=amp_dtype):
            total = torch.tensor(0.0, device=device)
            losses = {}
            pred_full = None
            pred_rgb = None

            if use_inr:
                coords, target_rgb = sample_coordinates(
                    target_frame,
                    config.train.num_samples,
                    target_time
                )
                pred_rgb = model(input_frames, input_times, coords)
                coord_losses = criterion(
                    pred_rgb, target_rgb,
                    model=model,
                    input_frames=input_frames,
                    input_times=input_times,
                    compute_all=False
                )
                for name, val in coord_losses.items():
                    losses[f"coord_{name}"] = val
                total = total + config.train.coordinate_loss_weight * coord_losses['total']

            if use_full:
                full_mode = 'nafnet' if getattr(model, 'use_nafnet_decoder', False) else 'fast'
                pred_full = model.render_frame(input_frames, input_times, target_time, mode=full_mode)
                full_losses = criterion(pred_full, target_frame, compute_all=True)
                for name, val in full_losses.items():
                    losses[f"full_{name}"] = val
                total = total + config.train.full_image_loss_weight * full_losses['total']

            if (distill_enabled and distill_weight > 0 and use_inr and use_full and
                    pred_full is not None and pred_rgb is not None and epoch >= distill_start_epoch):
                naf_rgb = sample_rgb_from_frame(pred_full, coords)
                diff = naf_rgb - pred_rgb
                distill = torch.sqrt(diff * diff + config.train.charbonnier_eps ** 2).mean()
                losses['distill'] = distill
                total = total + distill_weight * distill

            losses['total'] = total

        # Backward pass
        scaler.scale(losses['total']).backward()

        # Gradient clipping
        if config.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        for name, val in losses.items():
            if isinstance(val, torch.Tensor):
                total_losses[name] += val.item()
            elif name != '_weights':
                total_losses[name] += val
        full_ssim = None
        if use_full and pred_full is not None:
            mse = F.mse_loss(pred_full.float(), target_frame.float())
            full_ssim = _compute_ssim(pred_full.detach(), target_frame.detach())
        else:
            mse = F.mse_loss(pred_rgb.float(), target_rgb.float())
        total_psnr += (10 * torch.log10(1.0 / mse.clamp(min=1e-10))).item()
        if full_ssim is not None:
            total_ssim += full_ssim
        num_batches += 1

        # Progress bar
        postfix = {'loss': f"{losses['total'].item():.4f}"}
        if 'coord_charbonnier' in losses:
            postfix['charb'] = f"{losses['coord_charbonnier'].item():.4f}"
        if 'coord_anchor_distill' in losses:
            postfix['anchor'] = f"{losses['coord_anchor_distill'].item():.4f}"
        if full_ssim is not None:
            postfix['ssim'] = f"{full_ssim:.4f}"
        pbar.set_postfix(postfix)

        # Tensorboard logging
        global_step = epoch * len(dataloader) + batch_idx
        if writer and batch_idx % config.train.log_every == 0:
            for name, val in losses.items():
                if isinstance(val, torch.Tensor):
                    writer.add_scalar(f'train/{name}', val.item(), global_step)
            if full_ssim is not None:
                writer.add_scalar("train/ssim", full_ssim, global_step)

            # Log uncertainty weights
            if criterion.use_uncertainty_weighting:
                stats = criterion.get_uncertainty_stats()
                for name, weight in stats.get('weights', {}).items():
                    writer.add_scalar(f'uncertainty/weight_{name}', weight, global_step)
                for name, sigma in stats.get('sigmas', {}).items():
                    writer.add_scalar(f'uncertainty/sigma_{name}', sigma, global_step)

        # Periodic full-frame evaluation (for monitoring perceptual losses)
        if (config.train.full_frame_every > 0 and
            batch_idx > 0 and
            batch_idx % config.train.full_frame_every == 0):

            with torch.no_grad():
                full_mode = 'nafnet' if getattr(model, 'use_nafnet_decoder', False) else 'fast'
                pred_frame = model.render_frame(
                    input_frames, input_times,
                    target_time=target_time[0].item(),
                    mode=full_mode
                )
                full_losses = criterion(pred_frame, target_frame, compute_all=True)

                if writer:
                    for name, val in full_losses.items():
                        if isinstance(val, torch.Tensor) and name != 'total':
                            writer.add_scalar(f'train_ff/{name}', val.item(), global_step)

    metrics = {name: val / max(num_batches, 1) for name, val in total_losses.items()}
    metrics['loss'] = metrics.get('total', 0.0)
    metrics['psnr'] = total_psnr / max(num_batches, 1)
    metrics['ssim'] = total_ssim / max(num_batches, 1)
    return metrics


@torch.no_grad()
def validate(model: SPACE,
             dataloader,
             criterion: SPACELoss,
             config,
             epoch: int,
             writer: SummaryWriter = None) -> dict:
    """Validate model with full frame rendering."""
    model.eval()
    device = next(model.parameters()).device

    total_psnr = 0
    total_ssim = 0
    total_loss = 0
    num_samples = 0

    for input_frames, input_times, target_frame, target_time in tqdm(dataloader, desc="Validating"):
        input_frames = input_frames.to(device)
        input_times = input_times.to(device)
        target_frame = target_frame.to(device)

        if isinstance(target_time, (int, float)):
            target_time_val = target_time
        else:
            target_time_val = target_time[0].item()

        B, _, H, W = target_frame.shape

        # Render full frame (prefer NAFNet if available)
        full_mode = 'nafnet' if getattr(model, 'use_nafnet_decoder', False) else 'fast'
        pred_frame = model.render_frame(
            input_frames, input_times,
            target_time=target_time_val,
            mode=full_mode
        )

        # Compute losses (full image mode)
        losses = criterion(pred_frame, target_frame, compute_all=True)
        total_loss += losses['total'].item() * B

        # Compute PSNR
        mse = F.mse_loss(pred_frame, target_frame)
        psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))

        # SSIM from loss (1 - ssim_loss = ssim)
        ssim = 1.0 - losses['ssim'].item()

        total_psnr += psnr.item() * B
        total_ssim += ssim * B
        num_samples += B

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_loss = total_loss / num_samples

    if writer:
        writer.add_scalar('val/psnr', avg_psnr, epoch)
        writer.add_scalar('val/ssim', avg_ssim, epoch)
        writer.add_scalar('val/loss', avg_loss, epoch)

        # Log sample images
        if input_frames.shape[0] > 0:
            writer.add_images('val/input_0', input_frames[:4, 0], epoch)
            writer.add_images('val/target', target_frame[:4], epoch)
            writer.add_images('val/prediction', pred_frame[:4].clamp(0, 1), epoch)

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'loss': avg_loss,
    }


def save_checkpoint(model: SPACE,
                    optimizer,
                    scheduler,
                    scaler: amp.GradScaler,
                    epoch: int,
                    metrics: dict,
                    config,
                    path: str,
                    criterion=None):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
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
    # Save criterion state if it has learnable parameters (uncertainty weights)
    if criterion is not None and hasattr(criterion, 'state_dict'):
        checkpoint['criterion_state_dict'] = criterion.state_dict()
    torch.save(checkpoint, path)


def main():
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

    parser = argparse.ArgumentParser(description="Train SPACE model")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    parser.add_argument('--fast', action='store_true', help="Use fast/tiny config for testing")
    parser.add_argument('--mixed', action='store_true', help="Use mixed Vimeo+X4K training")
    parser.add_argument('--no-uncertainty', action='store_true', help="Disable uncertainty weighting (fixed loss weights)")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument('--device', type=str, default='cuda', help="Device to train on")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.load(args.config)
    elif args.fast:
        config = get_fast_config()
    elif args.mixed:
        config = get_mixed_config()
    elif getattr(args, 'no_uncertainty', False):
        config = get_no_uncertainty_config()
    else:
        config = get_default_config()

    # Override device
    config.train.device = args.device

    # Set seed
    set_seed(config.train.seed)

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.name}_{timestamp}"
    checkpoint_dir = Path(config.train.checkpoint_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(checkpoint_dir / "config.json")

    # Setup logging
    writer = SummaryWriter(checkpoint_dir / "logs")

    # Create model
    print("Creating model...")
    model = create_model(config)

    # Enable gradient checkpointing
    if config.train.use_gradient_checkpointing:
        model.set_gradient_checkpointing(True)
        print("  Gradient checkpointing: enabled")

    model = model.to(config.train.device)

    # Apply torch.compile
    if config.train.use_compile:
        print(f"  Compiling model with mode='{config.train.compile_mode}'...")
        model = torch.compile(model, mode=config.train.compile_mode)
        print("  Model compiled successfully")

    # Print model info
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    param_counts = base_model.get_param_count()
    print(f"Model parameters: {param_counts['total']:,}")
    for name, count in param_counts.items():
        if name != 'total':
            print(f"  {name}: {count:,}")

    # Create loss function
    if config.train.use_advanced_loss:
        print("Using advanced loss system with uncertainty weighting...")
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
        criterion = SPACELoss(
            charbonnier_weight=config.train.charbonnier_weight,
            charbonnier_eps=config.train.charbonnier_eps,
            ssim_weight=config.train.ssim_weight,
            perceptual_weight=config.train.perceptual_weight,
            frequency_weight=config.train.frequency_weight,
            use_perceptual=config.train.use_perceptual,
        )
    criterion = criterion.to(config.train.device)

    # Create dataloaders
    print("Creating dataloaders...")
    if config.data.use_mixed_training:
        print("Using mixed Vimeo + X4K training...")
        train_loader = get_mixed_dataloader(config, split="train")
        val_loader = get_mixed_dataloader(config, split="test")
    else:
        train_loader = get_dataloader(config, split="train")
        val_loader = get_dataloader(config, split="test")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create optimizer (include uncertainty params if using advanced loss)
    if config.train.use_advanced_loss and config.train.use_uncertainty_weighting:
        # Separate learning rate for uncertainty parameters (slower learning)
        optimizer = torch.optim.AdamW([
            {'params': model.parameters(), 'lr': config.train.learning_rate},
            {'params': criterion.uncertainty.parameters(), 'lr': config.train.learning_rate * 0.1},
        ], weight_decay=config.train.weight_decay)
        print(f"  Uncertainty params added to optimizer (lr={config.train.learning_rate * 0.1:.2e})")
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

    # Mixed precision scaler
    scaler = amp.GradScaler('cuda', enabled=config.train.use_amp)

    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.train.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        # Load criterion state (uncertainty weights) if available
        if 'criterion_state_dict' in checkpoint and hasattr(criterion, 'load_state_dict'):
            criterion.load_state_dict(checkpoint['criterion_state_dict'])
            print("  Loaded criterion state (uncertainty weights)")
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['metrics'].get('psnr', 0)

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    use_advanced = config.train.use_advanced_loss
    for epoch in range(start_epoch, config.train.epochs):
        # Train (use advanced training loop if advanced loss is enabled)
        if use_advanced:
            train_metrics = train_epoch_advanced(
                model, train_loader, optimizer, scaler, criterion,
                config, epoch, writer
            )
        else:
            train_metrics = train_epoch(
                model, train_loader, optimizer, scaler, criterion,
                config, epoch, writer
            )

        # Update scheduler
        if scheduler:
            scheduler.step()
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        # Validate
        if epoch % config.train.val_every == 0:
            val_metrics = validate(model, val_loader, criterion, config, epoch, writer)
            print(f"Epoch {epoch}: Loss={train_metrics['loss']:.4f}, "
                  f"PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.4f}")

            # Save best model
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, val_metrics, config,
                    checkpoint_dir / "best.pt",
                    criterion=criterion
                )
        else:
            print(f"Epoch {epoch}: Loss={train_metrics['loss']:.4f}")

        # Save periodic checkpoint
        if epoch % config.train.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, train_metrics, config,
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                criterion=criterion
            )

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        config.train.epochs - 1, train_metrics, config,
        checkpoint_dir / "final.pt",
        criterion=criterion
    )

    print(f"\nTraining complete! Best PSNR: {best_psnr:.2f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    writer.close()


if __name__ == "__main__":
    main()
