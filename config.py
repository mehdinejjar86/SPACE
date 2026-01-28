# config.py
# Configuration for SPACE training and model

from dataclasses import dataclass, field
from typing import Optional, List, Union
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """SPACE model configuration."""
    # Encoder (ConvNeXt)
    encoder_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    encoder_depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])

    # Latent/hidden dimensions
    latent_dim: int = 256
    hidden_dim: int = 256

    # Attention
    num_heads: int = 8

    # SIREN decoder
    num_siren_layers: int = 4
    omega_0: float = 30.0

    # Features
    use_feature_grids: bool = True
    use_fast_decoder: bool = False

    # Coarse-to-Fine Refinement
    use_refinement: bool = False
    refinement_hidden_dim: int = 64
    refinement_num_blocks: int = 4
    refinement_iterations: int = 1

    # NAFNet-style decoder (alternative to fast_decoder)
    use_nafnet_decoder: bool = True
    nafnet_hidden_dim: int = 64
    nafnet_num_blocks: List[int] = field(default_factory=lambda: [2, 2, 4, 4])
    nafnet_use_modulation: bool = True

    # Preset config (overrides above if set)
    preset: Optional[str] = None  # 'tiny', 'base', 'large'


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_root: str = "./data/vimeo_septuplet"
    train_list: str = "sep_trainlist.txt"
    test_list: str = "sep_testlist.txt"

    # Number of input frames (temporal views)
    num_input_frames: int = 4

    # Which frames to use as input (indices into 7-frame sequence)
    # Default: evenly spaced [0, 2, 4, 6]
    input_frame_indices: List[int] = field(default_factory=lambda: [0, 2, 4, 6])

    # Which frames can be targets (indices into 7-frame sequence)
    # Default: intermediate frames [1, 3, 5]
    target_frame_indices: List[int] = field(default_factory=lambda: [1, 3, 5])

    # Image size
    crop_size: int = 256
    resize_height: int = 256
    resize_width: int = 448

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

    # Mixed training (Vimeo + X4K)
    use_mixed_training: bool = False
    vimeo_root: str = "./data/vimeo_triplet"
    vimeo_mode: str = "mix"  # "interp", "extrap_fwd", "extrap_bwd", "mix"
    x4k_root: str = "./data/x4k1000"
    x4k_steps: List[int] = field(default_factory=lambda: [5, 31, 31])
    x4k_n_frames: List[int] = field(default_factory=lambda: [4, 3, 2])  # Paired with x4k_steps
    x4k_crop_size: int = 512
    dataset_ratios: List[float] = field(default_factory=lambda: [0.7, 0.3])  # [vimeo, x4k]

    # Validation datasets (for dual-dataset validation)
    vimeo_val_root: Optional[str] = None  # Vimeo triplet for validation
    x4k_val_root: Optional[str] = None    # X4K for validation


@dataclass
class TrainConfig:
    """Training configuration."""
    # Optimization
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 300

    # Learning rate schedule
    scheduler: str = "cosine"  # "cosine", "step", "none"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Training mode
    training_mode: str = "hybrid"  # "coordinate" (INR), "full_image" (NAFNet), "hybrid" (both)
    coordinate_loss_weight: float = 1.0
    full_image_loss_weight: float = 1.0

    # Curriculum (first fraction of training)
    curriculum_enabled: bool = True
    curriculum_fraction: float = 0.2

    # Distillation (INR -> NAFNet) during training
    distill_enabled: bool = True
    distill_weight: float = 0.1
    distill_start_epoch: int = 0

    # Coordinate sampling for INR training (only used when training_mode="coordinate")
    num_samples: int = 4096  # Coordinates per image per batch
    sample_strategy: str = "random"  # "random", "grid", "importance"

    # Loss weights (for SPACELoss - original)
    charbonnier_weight: float = 1.0
    charbonnier_eps: float = 1e-3  # Epsilon for Charbonnier loss (1e-3 better for PSNR than 1e-6)
    ssim_weight: float = 0.5
    perceptual_weight: float = 0.1
    frequency_weight: float = 0.1
    use_perceptual: bool = True  # Use VGG perceptual loss

    # Advanced Loss System (AdvancedSPACELoss) - default loss system
    use_advanced_loss: bool = True  # Use AdvancedSPACELoss (default)
    use_uncertainty_weighting: bool = True  # Learnable loss weights (can be disabled)

    # Advanced loss components
    use_msssim: bool = True       # Multi-scale SSIM
    use_lpips: bool = True        # LPIPS perceptual (AlexNet)
    use_gradient: bool = True     # Gradient/edge loss
    use_laplacian: bool = False   # Laplacian pyramid (heavy)
    use_frequency_adv: bool = True  # Frequency loss

    # Anchor distillation (self-consistency at input frame times)
    use_anchor_distillation: bool = True
    anchor_distill_weight: float = 0.5
    anchor_distill_indices: str = "all"  # "all" or "first_last"
    anchor_distill_samples: int = 512

    # Progressive curriculum
    use_progressive_loss: bool = True

    # LPIPS backbone
    lpips_net: str = "alex"  # "alex", "vgg", "squeeze"

    # Full-frame evaluation during training
    full_frame_every: int = 100  # Compute full-frame losses every N batches

    # Gradient clipping
    grad_clip: float = 1.0

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16", "bfloat16"

    # Optimization
    use_compile: bool = False  # torch.compile() for faster training
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    use_gradient_checkpointing: bool = False  # Reduce memory at cost of speed

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10  # Save every N epochs
    keep_last: int = 3  # Keep last N checkpoints

    # Logging
    log_every: int = 100  # Log every N steps
    val_every: int = 1  # Validate every N epochs
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"
    num_gpus: int = 1  # For DDP


@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Experiment name
    name: str = "space_default"

    def save(self, path: str):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        config_dict = {
            'name': self.name,
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'train': self.train.__dict__,
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        config = cls(
            name=config_dict.get('name', 'space_default'),
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            train=TrainConfig(**config_dict.get('train', {})),
        )
        return config


def get_default_config() -> Config:
    """Get default configuration (base model)."""
    return Config(
        name="space_base",
        model=ModelConfig(preset='base'),
    )


def get_tiny_config() -> Config:
    """Get tiny configuration for fast experimentation."""
    return Config(
        name="space_tiny",
        model=ModelConfig(preset='tiny', use_nafnet_decoder=True, use_fast_decoder=False),
        data=DataConfig(
            crop_size=128,
            num_workers=2,
        ),
        train=TrainConfig(
            batch_size=4,
            epochs=50,
            num_samples=1024,
            log_every=50,
            training_mode="hybrid",
            # Disable heavy losses for speed
            use_lpips=False,
            use_laplacian=False,
            use_uncertainty_weighting=False,
        ),
    )


def get_fast_config() -> Config:
    """Alias for get_tiny_config."""
    return get_tiny_config()


def get_large_config() -> Config:
    """Get configuration for best quality training."""
    return Config(
        name="space_large",
        model=ModelConfig(preset='large'),
        data=DataConfig(
            crop_size=256,
            num_workers=8,
        ),
        train=TrainConfig(
            batch_size=4,  # Smaller batch for larger model
            epochs=300,
            num_samples=8192,
            use_amp=True,
        ),
    )


def get_mixed_config() -> Config:
    """Get configuration for mixed Vimeo + X4K training.

    Uses TEMPO-style X4K configuration:
    - step=5 with N=4: small motion, more temporal context
    - step=31 with N=3: large motion, medium context
    - step=31 with N=2: large motion, minimal context
    """
    return Config(
        name="space_mixed",
        model=ModelConfig(preset='base'),
        data=DataConfig(
            use_mixed_training=True,
            vimeo_root="./data/vimeo_triplet",
            vimeo_mode="mix",
            x4k_root="./data/x4k1000",
            x4k_steps=[5, 31, 31],
            x4k_n_frames=[4, 3, 2],
            x4k_crop_size=512,
            dataset_ratios=[0.7, 0.3],
            crop_size=256,
            num_workers=8,
        ),
        train=TrainConfig(
            batch_size=8,
            epochs=300,
            num_samples=4096,
            use_amp=True,
        ),
    )


def get_advanced_loss_config() -> Config:
    """Alias for get_default_config (advanced loss is now default)."""
    return get_default_config()


def get_no_uncertainty_config() -> Config:
    """Get configuration with advanced loss but WITHOUT uncertainty weighting (fixed weights)."""
    return Config(
        name="space_no_uncertainty",
        model=ModelConfig(preset='base'),
        data=DataConfig(
            crop_size=256,
            num_workers=8,
        ),
        train=TrainConfig(
            batch_size=8,
            epochs=300,
            num_samples=4096,
            use_amp=True,
            # Disable uncertainty weighting (use fixed weights)
            use_uncertainty_weighting=False,
        ),
    )
