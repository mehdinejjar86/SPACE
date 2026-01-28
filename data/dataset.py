# dataset.py
# Dataset loader for SPACE training

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SpaceDataset(Dataset):
    """
    Dataset for SPACE training.

    Loads video sequences and provides:
    - N input frames at known timestamps
    - Target frame at a random intermediate timestamp

    Designed for Vimeo-90K Septuplet dataset (7 frames per sequence),
    but can be adapted to other multi-frame datasets.
    """

    def __init__(self,
                 root: str,
                 split: str = "train",
                 split_file: Optional[str] = None,
                 input_indices: List[int] = None,
                 target_indices: List[int] = None,
                 crop_size: int = 256,
                 augment: bool = True):
        """
        Args:
            root: Root directory of dataset (e.g., vimeo_septuplet/sequences)
            split: "train" or "test"
            split_file: Path to split file listing sequences
            input_indices: Which frames to use as input (default: [0, 2, 4, 6])
            target_indices: Which frames can be targets (default: [1, 3, 5])
            crop_size: Size for random crops during training
            augment: Whether to apply data augmentation
        """
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == "train")

        # Default frame indices for 7-frame sequences
        self.input_indices = input_indices or [0, 2, 4, 6]
        self.target_indices = target_indices or [1, 3, 5]
        self.num_frames = 7  # Total frames in sequence

        # Load sequence list
        self.sequences = self._load_sequences(split_file)

        # Transforms
        self.to_tensor = transforms.ToTensor()

    def _load_sequences(self, split_file: Optional[str]) -> List[str]:
        """Load list of sequence paths."""
        sequences = []

        if split_file and os.path.exists(split_file):
            # Load from split file
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seq_path = self.root / line
                        if seq_path.exists():
                            sequences.append(str(seq_path))
        else:
            # Auto-discover sequences
            # Expected structure: root/category/sequence/
            for category in sorted(self.root.iterdir()):
                if category.is_dir():
                    for seq in sorted(category.iterdir()):
                        if seq.is_dir():
                            # Check if it has the expected frames
                            if (seq / "im1.png").exists():
                                sequences.append(str(seq))

        if len(sequences) == 0:
            raise ValueError(f"No sequences found in {self.root}")

        return sequences

    def _load_frame(self, seq_path: str, frame_idx: int) -> Image.Image:
        """Load a single frame from sequence."""
        # Vimeo uses 1-indexed filenames (im1.png to im7.png)
        frame_path = Path(seq_path) / f"im{frame_idx + 1}.png"
        return Image.open(frame_path).convert('RGB')

    def _random_crop(self, images: List[Image.Image]) -> List[Image.Image]:
        """Apply same random crop to all images."""
        w, h = images[0].size

        if w < self.crop_size or h < self.crop_size:
            # Resize if too small
            scale = max(self.crop_size / w, self.crop_size / h) * 1.1
            new_w, new_h = int(w * scale), int(h * scale)
            images = [img.resize((new_w, new_h), Image.BILINEAR) for img in images]
            w, h = new_w, new_h

        # Random crop position
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)

        return [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in images]

    def _augment(self, images: List[Image.Image]) -> List[Image.Image]:
        """Apply data augmentation (same to all images)."""
        # Random horizontal flip
        if random.random() > 0.5:
            images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]

        # Random vertical flip
        if random.random() > 0.5:
            images = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in images]

        # Random 90-degree rotation
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            images = [img.rotate(90 * k) for img in images]

        return images

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            input_frames: [N, 3, H, W] - N input frames
            input_times: [N] - timestamps of input frames (normalized to [0, 1])
            target_frame: [3, H, W] - target frame to reconstruct
            target_time: float - timestamp of target frame
        """
        seq_path = self.sequences[idx]

        # Load all frames
        all_frames = [self._load_frame(seq_path, i) for i in range(self.num_frames)]

        # Select target frame
        target_idx = random.choice(self.target_indices)
        target_frame = all_frames[target_idx]

        # Select input frames
        input_frames = [all_frames[i] for i in self.input_indices]

        # Combine for joint processing
        all_images = input_frames + [target_frame]

        # Random crop (same for all)
        if self.crop_size > 0:
            all_images = self._random_crop(all_images)

        # Augmentation (same for all)
        if self.augment:
            all_images = self._augment(all_images)

        # Split back
        input_frames = all_images[:-1]
        target_frame = all_images[-1]

        # Convert to tensors
        input_tensors = torch.stack([self.to_tensor(f) for f in input_frames])
        target_tensor = self.to_tensor(target_frame)

        # Compute timestamps (normalized to [0, 1])
        # For 7-frame sequence: frame 0 -> t=0, frame 6 -> t=1
        input_times = torch.tensor([i / (self.num_frames - 1) for i in self.input_indices])
        target_time = target_idx / (self.num_frames - 1)

        return input_tensors, input_times, target_tensor, target_time


class SpaceDatasetTriplet(Dataset):
    """
    Simplified dataset for Vimeo-90K Triplet (3 frames).

    Uses frames 0 and 2 as input, frame 1 as target.
    """

    def __init__(self,
                 root: str,
                 split: str = "train",
                 split_file: Optional[str] = None,
                 crop_size: int = 256,
                 augment: bool = True):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == "train")

        self.sequences = self._load_sequences(split_file)
        self.to_tensor = transforms.ToTensor()

    def _load_sequences(self, split_file: Optional[str]) -> List[str]:
        sequences = []

        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seq_path = self.root / line
                        if seq_path.exists():
                            sequences.append(str(seq_path))
        else:
            for category in sorted(self.root.iterdir()):
                if category.is_dir():
                    for seq in sorted(category.iterdir()):
                        if seq.is_dir() and (seq / "im1.png").exists():
                            sequences.append(str(seq))

        return sequences

    def _load_frame(self, seq_path: str, frame_idx: int) -> Image.Image:
        frame_path = Path(seq_path) / f"im{frame_idx + 1}.png"
        return Image.open(frame_path).convert('RGB')

    def _random_crop(self, images: List[Image.Image]) -> List[Image.Image]:
        w, h = images[0].size
        if w < self.crop_size or h < self.crop_size:
            scale = max(self.crop_size / w, self.crop_size / h) * 1.1
            new_w, new_h = int(w * scale), int(h * scale)
            images = [img.resize((new_w, new_h), Image.BILINEAR) for img in images]
            w, h = new_w, new_h

        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        return [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in images]

    def _augment(self, images: List[Image.Image]) -> List[Image.Image]:
        if random.random() > 0.5:
            images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
        if random.random() > 0.5:
            images = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in images]
        return images

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq_path = self.sequences[idx]

        # Load triplet
        frame0 = self._load_frame(seq_path, 0)
        frame1 = self._load_frame(seq_path, 1)  # Target
        frame2 = self._load_frame(seq_path, 2)

        all_images = [frame0, frame2, frame1]  # Input, Input, Target

        if self.crop_size > 0:
            all_images = self._random_crop(all_images)
        if self.augment:
            all_images = self._augment(all_images)

        input_frames = torch.stack([self.to_tensor(all_images[0]),
                                     self.to_tensor(all_images[1])])
        target_frame = self.to_tensor(all_images[2])
        input_times = torch.tensor([0.0, 1.0])
        target_time = 0.5

        return input_frames, input_times, target_frame, target_time


def get_dataloader(config,
                   split: str = "train",
                   distributed: bool = False) -> DataLoader:
    """
    Create DataLoader from config.

    Args:
        config: DataConfig or full Config object
        split: "train" or "test"
        distributed: Whether to use DistributedSampler
    """
    # Handle full Config vs DataConfig
    if hasattr(config, 'data'):
        data_config = config.data
        train_config = config.train
    else:
        data_config = config
        train_config = None

    # Create dataset
    dataset = SpaceDataset(
        root=data_config.dataset_root,
        split=split,
        input_indices=data_config.input_frame_indices,
        target_indices=data_config.target_frame_indices,
        crop_size=data_config.crop_size if split == "train" else 0,
        augment=(split == "train"),
    )

    # Create sampler for distributed training
    sampler = None
    shuffle = (split == "train")
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    # Get batch size
    batch_size = train_config.batch_size if train_config else 8

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=(split == "train"),
    )
