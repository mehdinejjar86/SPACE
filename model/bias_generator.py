# bias_generator.py
# Hypernetwork that generates SIREN biases from scene representation

import torch
import torch.nn as nn


class BiasGenerator(nn.Module):
    """
    Generate per-layer biases for SIREN decoder from scene representation.

    This is the key mechanism that allows the network to adapt to different
    video content. Instead of having fixed weights, the SIREN biases are
    generated dynamically based on what the input frames "tell" us about
    the scene.

    Inspired by ActINR (CVPR 2025) - bias modulation for implicit representations.

    The generators are initialized to output near-zero biases, so initially
    the network behaves like a standard SIREN. Training learns what biases
    best reconstruct the scene.
    """

    def __init__(self,
                 scene_dim: int = 256,
                 hidden_dims: list = None):
        """
        Args:
            scene_dim: Dimension of scene representation from aggregator
            hidden_dims: List of dimensions for each SIREN layer
                        Default: [256, 256, 256, 3] for 4-layer SIREN
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 3]

        self.scene_dim = scene_dim
        self.hidden_dims = hidden_dims

        # Create a bias generator for each SIREN layer
        self.generators = nn.ModuleList()
        for dim in hidden_dims:
            gen = nn.Sequential(
                nn.Linear(scene_dim, scene_dim),
                nn.GELU(),
                nn.Linear(scene_dim, dim),
            )
            self.generators.append(gen)

        # Initialize to output near-zero biases
        self._init_zero_output()

    def _init_zero_output(self):
        """Initialize final layer of each generator to output near-zero."""
        for gen in self.generators:
            # Last linear layer
            final_layer = gen[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def forward(self, scene_code: torch.Tensor) -> list:
        """
        Args:
            scene_code: [B, scene_dim] scene representation

        Returns:
            List of [B, dim] bias tensors, one per SIREN layer
        """
        biases = []
        for gen in self.generators:
            bias = gen(scene_code)
            biases.append(bias)
        return biases


class BiasGeneratorWithScale(nn.Module):
    """
    Extended bias generator that also produces scale factors.

    Instead of just bias: output = Wx + b + bias_mod
    Also adds scale: output = scale_mod * (Wx + b) + bias_mod

    More expressive but also more parameters.
    """

    def __init__(self,
                 scene_dim: int = 256,
                 hidden_dims: list = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 3]

        self.scene_dim = scene_dim
        self.hidden_dims = hidden_dims

        # Bias generators
        self.bias_generators = nn.ModuleList()
        for dim in hidden_dims:
            gen = nn.Sequential(
                nn.Linear(scene_dim, scene_dim),
                nn.GELU(),
                nn.Linear(scene_dim, dim),
            )
            self.bias_generators.append(gen)

        # Scale generators
        self.scale_generators = nn.ModuleList()
        for dim in hidden_dims:
            gen = nn.Sequential(
                nn.Linear(scene_dim, scene_dim),
                nn.GELU(),
                nn.Linear(scene_dim, dim),
            )
            self.scale_generators.append(gen)

        self._init_weights()

    def _init_weights(self):
        """Initialize for identity operation at start."""
        for gen in self.bias_generators:
            nn.init.zeros_(gen[-1].weight)
            nn.init.zeros_(gen[-1].bias)

        for gen in self.scale_generators:
            nn.init.zeros_(gen[-1].weight)
            # Initialize bias to 1 for identity scaling
            nn.init.ones_(gen[-1].bias)

    def forward(self, scene_code: torch.Tensor):
        """
        Returns:
            biases: List of [B, dim] bias tensors
            scales: List of [B, dim] scale tensors
        """
        biases = [gen(scene_code) for gen in self.bias_generators]
        scales = [gen(scene_code) for gen in self.scale_generators]
        return biases, scales


class BiasGeneratorHierarchical(nn.Module):
    """
    Hierarchical bias generator with shared trunk.

    More parameter-efficient: shared feature extraction,
    per-layer heads for bias generation.
    """

    def __init__(self,
                 scene_dim: int = 256,
                 hidden_dims: list = None,
                 trunk_depth: int = 2):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 3]

        # Shared trunk
        trunk_layers = []
        for _ in range(trunk_depth):
            trunk_layers.extend([
                nn.Linear(scene_dim, scene_dim),
                nn.GELU(),
            ])
        self.trunk = nn.Sequential(*trunk_layers)

        # Per-layer heads
        self.heads = nn.ModuleList([
            nn.Linear(scene_dim, dim) for dim in hidden_dims
        ])

        # Zero-init heads
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, scene_code: torch.Tensor) -> list:
        """
        Args:
            scene_code: [B, scene_dim]

        Returns:
            List of [B, dim] bias tensors
        """
        features = self.trunk(scene_code)
        return [head(features) for head in self.heads]
