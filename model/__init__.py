# SPACE: Spacetime Implicit Neural Representation
# Treats video as a continuous 3D volume queryable at any (x, y, t)

# Main SPACE model (enhanced with ConvNeXt, multi-scale features, hybrid decoder)
from .space import SPACE, build_space

# Core components
from .encoder_x import FrameEncoderX, ConvNeXtEncoder
from .aggregator_x import TemporalAggregatorX
from .decoder_x import HybridDecoder, UpsampleDecoder, FourierFeatures
from .bias_generator import BiasGenerator
from .refinement import RefinementModule, CoarseToFineDecoder

# Loss functions (original)
from .loss_x import SPACELoss, VGGPerceptualLoss, CharbonnierLoss, SSIMLoss, FrequencyLoss

# Advanced loss system
from .loss_advanced import (
    AdvancedSPACELoss,
    UncertaintyWeighting,
    LPIPSLoss,
    MultiScaleSSIMLoss,
    GradientLoss,
    LaplacianPyramidLoss,
    CoordinateAnchorDistillationLoss,
    ProgressiveLossScheduler,
    FullFrameAnchorDistillationLoss,
    HybridLoss,
)

__all__ = [
    # Main model
    'SPACE',
    'build_space',
    # Encoder
    'FrameEncoderX',
    'ConvNeXtEncoder',
    # Aggregator
    'TemporalAggregatorX',
    # Decoder
    'HybridDecoder',
    'UpsampleDecoder',
    'FourierFeatures',
    # Bias generator
    'BiasGenerator',
    # Refinement
    'RefinementModule',
    'CoarseToFineDecoder',
    # Original losses
    'SPACELoss',
    'VGGPerceptualLoss',
    'CharbonnierLoss',
    'SSIMLoss',
    'FrequencyLoss',
    # Advanced loss system
    'AdvancedSPACELoss',
    'UncertaintyWeighting',
    'LPIPSLoss',
    'MultiScaleSSIMLoss',
    'GradientLoss',
    'LaplacianPyramidLoss',
    'CoordinateAnchorDistillationLoss',
    'ProgressiveLossScheduler',
    'FullFrameAnchorDistillationLoss',
    'HybridLoss',
]
