#!/usr/bin/env python3
# inference.py
# Inference utilities for SPACE model
# Output resolution always matches input (maintains creator intent)

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from model import SPACE, build_space
from config import Config


def load_model(checkpoint_path: str, device: str = 'cuda') -> SPACE:
    """Load trained SPACE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config
    model_config = checkpoint['config']['model']

    # Check if using preset
    if model_config.get('preset'):
        model = build_space(model_config['preset'])
    else:
        model = SPACE(
            encoder_dims=model_config.get('encoder_dims', [64, 128, 256, 512]),
            encoder_depths=model_config.get('encoder_depths', [2, 2, 6, 2]),
            latent_dim=model_config.get('latent_dim', 256),
            num_heads=model_config.get('num_heads', 8),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_siren_layers=model_config.get('num_siren_layers', 4),
            omega_0=model_config.get('omega_0', 30.0),
            use_feature_grids=model_config.get('use_feature_grids', True),
            use_fast_decoder=model_config.get('use_fast_decoder', True),
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def load_frames(frame_paths: list, device: str = 'cuda') -> torch.Tensor:
    """Load frames from file paths."""
    to_tensor = transforms.ToTensor()

    frames = []
    for path in frame_paths:
        img = Image.open(path).convert('RGB')
        frames.append(to_tensor(img))

    # Stack: [N, 3, H, W] -> [1, N, 3, H, W]
    return torch.stack(frames).unsqueeze(0).to(device)


def save_frame(tensor: torch.Tensor, path: str):
    """Save frame tensor to file."""
    # tensor: [1, 3, H, W] or [3, H, W]
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Clamp to valid range
    tensor = tensor.clamp(0, 1)

    # Convert to PIL
    img = transforms.ToPILImage()(tensor.cpu())
    img.save(path)


@torch.no_grad()
def interpolate(model: SPACE,
                input_frames: torch.Tensor,
                input_times: torch.Tensor,
                target_time: float,
                mode: str = 'auto') -> torch.Tensor:
    """
    Interpolate a frame at target_time.

    Output resolution matches input frame resolution (no super-resolution).

    Args:
        model: Trained SPACE model
        input_frames: [1, N, 3, H, W] input frames
        input_times: [1, N] timestamps
        target_time: Target timestamp
        mode: 'auto', 'hq', or 'fast'

    Returns:
        [1, 3, H, W] interpolated frame at input resolution
    """
    return model.render_frame(
        input_frames, input_times,
        target_time=target_time,
        mode=mode
    )


@torch.no_grad()
def generate_video(model: SPACE,
                   input_frames: torch.Tensor,
                   input_times: torch.Tensor,
                   num_frames: int,
                   start_time: float = 0.0,
                   end_time: float = 1.0,
                   mode: str = 'fast') -> list:
    """
    Generate a sequence of interpolated frames.

    Output resolution matches input frame resolution.

    Args:
        model: Trained SPACE model
        input_frames: [1, N, 3, H, W] input frames
        input_times: [1, N] timestamps
        num_frames: Number of frames to generate
        start_time: Start timestamp
        end_time: End timestamp
        mode: Rendering mode ('auto', 'hq', 'fast')

    Returns:
        List of [1, 3, H, W] frame tensors at input resolution
    """
    times = torch.linspace(start_time, end_time, num_frames)

    frames = []
    for t in times:
        frame = interpolate(model, input_frames, input_times, t.item(), mode)
        frames.append(frame)

    return frames


def main():
    parser = argparse.ArgumentParser(description="SPACE Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--frames', type=str, nargs='+', required=True, help="Input frame paths")
    parser.add_argument('--times', type=float, nargs='+', required=True, help="Input frame timestamps")
    parser.add_argument('--target-time', type=float, required=True, help="Target timestamp")
    parser.add_argument('--output', type=str, required=True, help="Output path")
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'hq', 'fast'],
                        help="Rendering mode")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    args = parser.parse_args()

    # Validate inputs
    if len(args.frames) != len(args.times):
        raise ValueError("Number of frames must match number of timestamps")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)

    # Load frames
    print(f"Loading {len(args.frames)} input frames...")
    input_frames = load_frames(args.frames, args.device)
    input_times = torch.tensor([args.times], device=args.device)

    H, W = input_frames.shape[3], input_frames.shape[4]
    print(f"Input resolution: {H}x{W}")

    # Interpolate
    print(f"Interpolating frame at t={args.target_time} (mode={args.mode})...")
    output = interpolate(model, input_frames, input_times, args.target_time, args.mode)

    print(f"Output resolution: {output.shape[2]}x{output.shape[3]} (matches input)")

    # Save
    save_frame(output, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
