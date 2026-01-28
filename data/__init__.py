# SPACE data module
# Dataloaders for video frame interpolation training

# Original dataset (generic Vimeo-90K loader)
from .dataset import SpaceDataset, SpaceDatasetTriplet, get_dataloader

# Vimeo-90K with mix mode support (interp/extrap)
from .vimeo import VimeoTriplet, VimeoSeptuplet, vimeo_collate

# X4K1000 with STEP-based motion sampling and variable N
from .x4k import (
    X4K1000Dataset,
    x4k_collate,
    X4KBatchSampler,
    DistributedX4KBatchSampler,
    X4KTestDataset,
    x4k_test_collate,
    X4KSequenceDataset,
    x4k_sequence_collate,
)

# Samplers for mixed-dataset training
from .samplers import PureBatchSampler, DistributedPureBatchSampler

__all__ = [
    # Original loaders
    'SpaceDataset',
    'SpaceDatasetTriplet',
    'get_dataloader',
    # Vimeo with mix mode
    'VimeoTriplet',
    'VimeoSeptuplet',
    'vimeo_collate',
    # X4K with STEP sampling and variable N
    'X4K1000Dataset',
    'x4k_collate',
    'X4KBatchSampler',
    'DistributedX4KBatchSampler',
    'X4KTestDataset',
    'x4k_test_collate',
    'X4KSequenceDataset',
    'x4k_sequence_collate',
    # Samplers
    'PureBatchSampler',
    'DistributedPureBatchSampler',
]
