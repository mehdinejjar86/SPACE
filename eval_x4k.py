#!/usr/bin/env python3
"""
X4K1000FPS Evaluation Script for SPACE

Evaluates SPACE model on X4K test set using cascaded prediction:
- 8x interpolation: predict 7 frames between frame 0 and 32
- Hierarchical approach using variable N:
  1. N=2 (0, 32) → predict frame 16
  2. N=3 (0, 16, 32) → predict frames 8, 24
  3. N=4 (closest anchors) → predict frames 4, 12, 20, 28

Evaluation modes:
- XTEST-2k: Downscaled to 2048x1080
- XTEST-4k: Full 4K resolution
"""

import os
import sys
import cv2
import math
import glob
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from model import SPACE, build_space


def ssim_matlab(img1, img2):
    """Compute SSIM (MATLAB-compatible implementation)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    img1 = img1.float()
    img2 = img2.float()

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0).to(img1.device)

    mu1 = torch.nn.functional.conv2d(img1, window, padding=5, groups=1)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=5, groups=1)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 ** 2, window, padding=5, groups=1) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 ** 2, window, padding=5, groups=1) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=5, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def compute_psnr(pred, gt):
    """Compute PSNR between prediction and ground truth."""
    mse = torch.mean((pred - gt) ** 2)
    return -10 * math.log10(mse.cpu().item())


def compute_ssim(pred, gt):
    """Compute SSIM for RGB image (average across channels)."""
    ssim_vals = []
    for c in range(3):
        ssim_vals.append(ssim_matlab(pred[:, c:c+1], gt[:, c:c+1]).item())
    return np.mean(ssim_vals)


class InputPadder:
    """Pads images to be divisible by divisor."""

    def __init__(self, dims, divisor=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (divisor - (self.ht % divisor)) % divisor
        pad_wd = (divisor - (self.wd % divisor)) % divisor
        self._pad = [0, pad_wd, 0, pad_ht]

    def pad(self, *inputs):
        return [torch.nn.functional.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        return x[..., :self.ht, :self.wd]


def load_frame(path):
    """Load frame as numpy array [H, W, 3] in [0, 1]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return img.astype(np.float32) / 255.0


def numpy_to_tensor(img, device):
    """Convert numpy [H, W, 3] to tensor [1, 3, H, W]."""
    return torch.from_numpy(img.transpose(2, 0, 1)[None]).float().to(device)


def tensor_to_numpy(tensor):
    """Convert tensor [1, 3, H, W] to numpy [H, W, 3] in [0, 255]."""
    return (tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)


def get_x4k_test_sequences(data_path):
    """
    Get X4K test sequences.

    Returns:
        List of (type_name, seq_name, frame_paths) tuples
    """
    sequences = []

    for type_dir in sorted(glob.glob(os.path.join(data_path, 'Type*'))):
        type_name = os.path.basename(type_dir)
        for seq_dir in sorted(glob.glob(os.path.join(type_dir, '*'))):
            if not os.path.isdir(seq_dir):
                continue
            seq_name = os.path.basename(seq_dir)
            frames = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
            if len(frames) == 33:  # 0-32 frames
                sequences.append((type_name, seq_name, frames))

    return sequences


class SPACEPredictor:
    """
    SPACE model wrapper for X4K evaluation.

    Implements cascaded prediction using variable N:
    - Level 0: N=2 (endpoints) → predict middle
    - Level 1: N=3 (endpoints + middle) → predict quarters
    - Level 2: N=4 (closest 4 anchors) → predict eighths
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_frame(self, anchor_frames, anchor_indices, target_idx, total_frames=33):
        """
        Predict a single frame using closest anchors.

        Args:
            anchor_frames: dict {frame_idx: tensor [1, 3, H, W]}
            anchor_indices: list of available anchor indices
            target_idx: target frame index to predict
            total_frames: total frame count (33 for X4K test)

        Returns:
            Predicted frame tensor [1, 3, H, W]
        """
        # Sort anchor indices
        sorted_anchors = sorted(anchor_indices)

        # Find closest anchors to target
        # We want anchors that bracket the target (before and after)
        before = [i for i in sorted_anchors if i < target_idx]
        after = [i for i in sorted_anchors if i > target_idx]

        if not before or not after:
            raise ValueError(f"Target {target_idx} not bracketed by anchors {sorted_anchors}")

        # Select N anchors (up to 4, closest to target)
        # Strategy: always include the immediate neighbors, then expand
        selected = []

        # Add immediate neighbors
        selected.append(before[-1])  # Closest before
        selected.append(after[0])    # Closest after

        # Add more anchors if available (prefer closer ones)
        remaining_before = before[:-1][::-1]  # Reverse to get closest first
        remaining_after = after[1:]

        while len(selected) < 4 and (remaining_before or remaining_after):
            # Alternate between before and after
            if remaining_before:
                selected.append(remaining_before.pop(0))
            if len(selected) < 4 and remaining_after:
                selected.append(remaining_after.pop(0))

        selected = sorted(selected)
        N = len(selected)

        # Build input tensors
        frames_list = [anchor_frames[i] for i in selected]
        frames = torch.cat(frames_list, dim=0).unsqueeze(0)  # [1, N, 3, H, W]

        # Compute normalized times
        # Anchors span from first to last selected index
        first_idx, last_idx = selected[0], selected[-1]
        anchor_times = torch.tensor(
            [(i - first_idx) / (last_idx - first_idx) for i in selected],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # [1, N]

        target_time = (target_idx - first_idx) / (last_idx - first_idx)

        # Predict
        with torch.no_grad():
            pred = self.model.render_frame(
                frames, anchor_times,
                target_time=target_time,
                mode='fast'
            )

        return pred

    def predict_8x(self, frame_0, frame_32, padder=None):
        """
        Predict 7 intermediate frames using cascaded approach.

        Args:
            frame_0: First frame tensor [1, 3, H, W]
            frame_32: Last frame tensor [1, 3, H, W]
            padder: Optional InputPadder for unpadding

        Returns:
            dict {target_idx: predicted_frame} for indices 4, 8, 12, 16, 20, 24, 28
        """
        # Initialize anchor frames dict
        anchor_frames = {0: frame_0, 32: frame_32}
        predictions = {}

        # Level 0: N=2 (0, 32) → predict 16
        pred_16 = self._predict_with_n(
            anchor_frames, [0, 32], target_idx=16
        )
        anchor_frames[16] = pred_16
        predictions[16] = padder.unpad(pred_16) if padder else pred_16

        # Level 1: N=3 (0, 16, 32) → predict 8, 24
        for target_idx in [8, 24]:
            pred = self._predict_with_n(
                anchor_frames, [0, 16, 32], target_idx=target_idx
            )
            anchor_frames[target_idx] = pred
            predictions[target_idx] = padder.unpad(pred) if padder else pred

        # Level 2: N=4 (closest anchors) → predict 4, 12, 20, 28
        # Available anchors: 0, 8, 16, 24, 32
        level2_targets = [4, 12, 20, 28]
        for target_idx in level2_targets:
            pred = self.predict_frame(
                anchor_frames,
                list(anchor_frames.keys()),
                target_idx
            )
            predictions[target_idx] = padder.unpad(pred) if padder else pred

        return predictions

    def _predict_with_n(self, anchor_frames, anchor_indices, target_idx):
        """Predict using specific anchor indices."""
        N = len(anchor_indices)

        frames_list = [anchor_frames[i] for i in anchor_indices]
        frames = torch.cat(frames_list, dim=0).unsqueeze(0)  # [1, N, 3, H, W]

        first_idx, last_idx = anchor_indices[0], anchor_indices[-1]
        anchor_times = torch.tensor(
            [(i - first_idx) / (last_idx - first_idx) for i in anchor_indices],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        target_time = (target_idx - first_idx) / (last_idx - first_idx)

        with torch.no_grad():
            pred = self.model.render_frame(
                frames, anchor_times,
                target_time=target_time,
                mode='fast'
            )

        return pred


def evaluate_x4k(model, data_path, device, modes=['XTEST-2k', 'XTEST-4k']):
    """
    Evaluate SPACE on X4K1000FPS test set.

    Args:
        model: SPACE model
        data_path: Path to X4K test data (containing Type1/, Type2/, Type3/)
        device: torch device
        modes: List of evaluation modes ('XTEST-2k', 'XTEST-4k')

    Returns:
        dict with results per mode
    """
    sequences = get_x4k_test_sequences(data_path)
    print(f"Found {len(sequences)} test sequences")

    predictor = SPACEPredictor(model, device)
    results = {}

    # Target frame indices for 8x interpolation
    target_indices = [4, 8, 12, 16, 20, 24, 28]

    for mode in modes:
        print(f"\nEvaluating {mode}...")

        all_psnr = []
        all_ssim = []

        for type_name, seq_name, frames in tqdm(sequences, desc=mode):
            # Load anchor frames (0 and 32)
            frame_0_np = load_frame(frames[0])
            frame_32_np = load_frame(frames[32])

            # Resize for 2K mode
            if mode == 'XTEST-2k':
                frame_0_np = cv2.resize(frame_0_np, (2048, 1080), interpolation=cv2.INTER_AREA)
                frame_32_np = cv2.resize(frame_32_np, (2048, 1080), interpolation=cv2.INTER_AREA)

            # Convert to tensors
            frame_0 = numpy_to_tensor(frame_0_np, device)
            frame_32 = numpy_to_tensor(frame_32_np, device)

            # Pad for model
            padder = InputPadder(frame_0.shape, divisor=32)
            frame_0_pad, frame_32_pad = padder.pad(frame_0, frame_32)

            # Predict all intermediate frames
            predictions = predictor.predict_8x(frame_0_pad, frame_32_pad, padder)

            # Evaluate each target frame
            seq_psnr = []
            seq_ssim = []

            for target_idx in target_indices:
                # Load ground truth
                gt_np = load_frame(frames[target_idx])
                if mode == 'XTEST-2k':
                    gt_np = cv2.resize(gt_np, (2048, 1080), interpolation=cv2.INTER_AREA)
                gt = numpy_to_tensor(gt_np, device)

                # Get prediction (convert back through numpy for fair comparison)
                pred_np = tensor_to_numpy(predictions[target_idx])
                pred = numpy_to_tensor(pred_np / 255.0, device)

                # Compute metrics
                psnr = compute_psnr(pred, gt)
                ssim = compute_ssim(pred, gt)

                seq_psnr.append(psnr)
                seq_ssim.append(ssim)

            all_psnr.append(np.mean(seq_psnr))
            all_ssim.append(np.mean(seq_ssim))

        results[mode] = {
            'psnr': np.mean(all_psnr),
            'ssim': np.mean(all_ssim),
            'psnr_std': np.std(all_psnr),
            'ssim_std': np.std(all_ssim),
        }

        print(f"{mode}  PSNR: {results[mode]['psnr']:.4f}  SSIM: {results[mode]['ssim']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SPACE on X4K1000FPS')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to X4K test data (containing Type1/, Type2/, Type3/)')
    parser.add_argument('--preset', type=str, default='base',
                        choices=['tiny', 'base', 'large'],
                        help='Model preset')
    parser.add_argument('--mode', type=str, nargs='+',
                        default=['XTEST-2k', 'XTEST-4k'],
                        help='Evaluation modes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = build_space(args.preset)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Evaluate
    print(f"\n{'='*60}")
    print(f"X4K1000FPS Evaluation - SPACE ({args.preset})")
    print(f"{'='*60}")

    results = evaluate_x4k(model, args.path, device, modes=args.mode)

    print(f"\n{'='*60}")
    print("Final Results:")
    print(f"{'='*60}")
    for mode, metrics in results.items():
        print(f"{mode}:")
        print(f"  PSNR: {metrics['psnr']:.4f} +/- {metrics['psnr_std']:.4f}")
        print(f"  SSIM: {metrics['ssim']:.4f} +/- {metrics['ssim_std']:.4f}")


if __name__ == '__main__':
    main()
