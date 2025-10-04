"""
Spatial objective creation functions for Lucent neural network visualization.

This module provides comprehensive functions to create various types of spatial objectives:
- Center NxN arrays with offsets
- Corner, edge, and grid positioning
- Dual and custom spatial objectives
- Random objective generation with multiple types

Separated from parse_model_lucent.py for better code organization.
"""

import random
from lucent.optvis import objectives
from lucent_layer_utils import get_channels_from_lucent_name, get_layer_dimensions
from gabor_objectives import create_gabor_weighted_objective


def create_random_objective(
    model,
    layers_list,
    layer_name=None,
    sampled_channels=2,
    objective_types=None,
    offsets=None,
):
    """Create a random objective with valid channel index, optionally with a second offset objective
    Args:
        model: PyTorch model
        layers_list: List of layer names to choose from
        layer_name: Specific layer name to use (if None, picks random)
        objective_types: List of objective types - "channel", "neuron", "center_3x3", "center_5x5", "center_7x7", "gabor"
        sampled_channels: Number of channels to sample
        offsets: List of (x,y) offset tuples for additional objectives
    Returns:
        Combined objective or None if failed
    """
    # Set default values to avoid mutable defaults
    if objective_types is None:
        objective_types = ["neuron"]
    if offsets is None:
        offsets = []

    assert len(objective_types) == len(offsets) + 1
    objective_type = objective_types[0]

    # Pick a random layer
    if not layer_name:
        layer_name = random.choice(layers_list)

    # Get the number of channels for this layer
    num_channels = get_channels_from_lucent_name(model, layer_name)

    if num_channels is None:
        print(f"Could not determine channels for {layer_name}")
        return None

    height, width, _ = get_layer_dimensions(model, layer_name, input_size=384)

    objectives_list = []

    for n in range(sampled_channels):
        print(f"Objective {n} of {sampled_channels}")
        objectives_list.extend(
            [
                create_objective_for_layer(
                    layer_name,
                    (height, width),
                    objective_type,
                    num_channels,
                    with_offset=(0, 0),
                )
            ]
        )

    for objective_type, offset in zip(objective_types[1:], offsets):
        # Add second objective with offset if specified
        if objective_type is not None:
            for n in range(sampled_channels):
                print(f"Objective {n} of {sampled_channels}")
                objectives_list.extend(
                    [
                        create_objective_for_layer(
                            layer_name,
                            (height, width),
                            objective_type,
                            num_channels,
                            with_offset=offset,
                        )
                    ]
                )

    return sum(objectives_list)


def create_objective_for_layer(
    layer_name, layer_size, objective_type, num_channels, with_offset=(0, 0)
):
    """Create a specific type of objective for a layer.
    Args:
        layer_name: Name of the layer
        layer_size: (height, width) tuple of layer spatial dimensions
        objective_type: Type - "channel", "neuron", "center_3x3", "center_5x5", "center_7x7", "gabor"
        num_channels: Number of channels in the layer
        with_offset: (x, y) spatial offset tuple
    Returns:
        Lucent objective
    """
    # Pick a random valid channel index (0 to num_channels - 1)
    channel_idx = random.randint(0, num_channels - 1)

    print(f"Selected layer: {layer_name}")
    print(f"Available channels: 0-{num_channels-1}")
    print(f"Selected channel: {channel_idx}")

    # default to channel objective
    obj = objectives.channel(layer_name, channel_idx)

    if objective_type == "neuron":
        # For neuron objectives, we can't apply spatial offset, so use regular neuron
        obj = create_center_nxn_objective(
            layer_name,
            channel_idx,
            1,
            spatial_weight=random.uniform(-1.0, 1.0),
            with_offset=with_offset,
        )

    elif objective_type.startswith("center_") and "x" in objective_type:
        # Parse the size and create offset objective
        size_str = objective_type.replace("center_", "")
        size = int(size_str.split("x")[0])
        obj = create_center_nxn_objective(
            layer_name,
            channel_idx,
            size,
            spatial_weight=random.uniform(-1.0, 1.0),
            with_offset=with_offset,
        )
        print(f"✅ Created {size}x{size} offset objective at ({with_offset})")

    elif objective_type == "gabor":
        obj = create_gabor_weighted_objective(
            layer_name,
            channel_idx,
            size=layer_size[0],
            with_offset=with_offset,
            sigma=(random.uniform(0.5, 2.0), random.uniform(0.5, 2.0)),
            theta=random.uniform(0, 3.14),
            lambda_freq=random.uniform(1.5, 8.0),
            psi=random.uniform(0, 3.14),
            gamma=random.uniform(0.5, 1.0),
            normalize_weights=True,
        )
        print(f"✅ Created gabor objective at ({with_offset})")

    elif objective_type != "channel":
        raise ValueError(f"Invalid objective type: {objective_type}")

    return obj


def create_center_nxn_objective(
    layer_name, channel_idx, size=3, spatial_weight=1.0, with_offset=(0, 0)
):
    """
    Create an objective that targets an NxN array of neurons at a specified position in a feature map.
    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        size: Size of the region (e.g., 3 for 3x3, 5 for 5x5, 7 for 7x7)
        spatial_weight: Weight for the spatial averaging (default 1.0)
        with_offset: (x_offset, y_offset) tuple - horizontal and vertical offset from center
                    (positive x = right, negative x = left, positive y = down, negative y = up)
    Returns:
        Lucent objective targeting NxN region at specified offset
    """
    # Ensure size is odd for symmetric centering
    if size % 2 == 0:
        print(
            f"Warning: Even size {size} provided, using {size+1} for symmetric centering"
        )
        size += 1

    # Calculate the radius (how many pixels from center)
    radius = size // 2

    # Create a custom objective that targets the offset NxN region
    def offset_nxn_obj(model):
        # Get the activations for the specified layer and channel
        layer_acts = model(layer_name)

        if len(layer_acts.shape) != 4:  # Should be [batch, channels, height, width]
            print(
                f"Warning: Layer {layer_name} has unexpected shape {layer_acts.shape}"
            )
            return -layer_acts[:, channel_idx].mean()

        # Get spatial dimensions
        _, _, h, w = layer_acts.shape

        # Calculate center coordinates
        center_h, center_w = h // 2, w // 2

        # Apply offsets
        target_h = center_h + with_offset[1]
        target_w = center_w + with_offset[0]

        # Define NxN region around target position (with bounds checking)
        h_start = max(0, target_h - radius)
        h_end = min(h, target_h + radius + 1)
        w_start = max(0, target_w - radius)
        w_end = min(w, target_w + radius + 1)

        # Check if the offset region is valid (not entirely out of bounds)
        if h_start >= h or w_start >= w or h_end <= 0 or w_end <= 0:
            print(
                f"Warning: Offset region ({with_offset[0]}, {with_offset[1]}) "
                f"is out of bounds for {h}x{w} feature map"
            )
            # Fallback to center region
            h_start = max(0, center_h - radius)
            h_end = min(h, center_h + radius + 1)
            w_start = max(0, center_w - radius)
            w_end = min(w, center_w + radius + 1)

        # Extract NxN offset region
        offset_region = layer_acts[:, channel_idx, h_start:h_end, w_start:w_end]

        # Return negative mean (Lucent maximizes by minimizing negative)
        return -offset_region.mean() * spatial_weight

    # Create the objective using Lucent's Objective class
    return objectives.Objective(offset_nxn_obj)


def create_center_3x3_objective(layer_name, channel_idx, spatial_weight=1.0):
    """
    Backward compatibility wrapper for create_center_nxn_objective with size=3.
    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        spatial_weight: Weight for the spatial averaging (default 1.0)
    Returns:
        Lucent objective targeting 3x3 center neurons
    """
    return create_center_nxn_objective(
        layer_name, channel_idx, size=3, spatial_weight=spatial_weight
    )


def create_center_5x5_objective(layer_name, channel_idx, spatial_weight=1.0):
    """
    Convenience function for creating 5x5 center objectives.
    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        spatial_weight: Weight for the spatial averaging (default 1.0)
    Returns:
        Lucent objective targeting 5x5 center neurons
    """
    return create_center_nxn_objective(
        layer_name, channel_idx, size=5, spatial_weight=spatial_weight
    )


def create_center_7x7_objective(layer_name, channel_idx, spatial_weight=1.0):
    """
    Convenience function for creating 7x7 center objectives.
    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        spatial_weight: Weight for the spatial averaging (default 1.0)
    Returns:
        Lucent objective targeting 7x7 center neurons
    """
    return create_center_nxn_objective(
        layer_name, channel_idx, size=7, spatial_weight=spatial_weight
    )


def create_corner_objectives(layer_name, channel_idx, size=3, spatial_weight=1.0):
    """
    Create objectives for all four corners of a feature map.
    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        size: Size of the region (default 3x3)
        spatial_weight: Weight for the spatial averaging (default 1.0)
    Returns:
        Dict with objectives for 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    """
    # Calculate large offsets to reach corners (will be clamped by bounds checking)
    large_offset = 50  # Arbitrary large number

    return {
        "top_left": create_center_nxn_objective(
            layer_name,
            channel_idx,
            size,
            spatial_weight,
            with_offset=(-large_offset, -large_offset),
        ),
        "top_right": create_center_nxn_objective(
            layer_name,
            channel_idx,
            size,
            spatial_weight,
            with_offset=(large_offset, -large_offset),
        ),
        "bottom_left": create_center_nxn_objective(
            layer_name,
            channel_idx,
            size,
            spatial_weight,
            with_offset=(-large_offset, large_offset),
        ),
        "bottom_right": create_center_nxn_objective(
            layer_name,
            channel_idx,
            size,
            spatial_weight,
            with_offset=(large_offset, large_offset),
        ),
    }


def create_edge_objectives(
    layer_name, channel_idx, size=3, spatial_weight=1.0, edge_offset=10
):
    """
    Create objectives for the edges (top, bottom, left, right) of a feature map.
    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        size: Size of the region (default 3x3)
        spatial_weight: Weight for the spatial averaging (default 1.0)
        edge_offset: How far from center to place the edge objectives
    Returns:
        Dict with objectives for 'top', 'bottom', 'left', 'right'
    """
    return {
        "top": create_center_nxn_objective(
            layer_name, channel_idx, size, spatial_weight, with_offset=(0, -edge_offset)
        ),
        "bottom": create_center_nxn_objective(
            layer_name, channel_idx, size, spatial_weight, with_offset=(0, edge_offset)
        ),
        "left": create_center_nxn_objective(
            layer_name, channel_idx, size, spatial_weight, with_offset=(-edge_offset, 0)
        ),
        "right": create_center_nxn_objective(
            layer_name, channel_idx, size, spatial_weight, with_offset=(edge_offset, 0)
        ),
    }


def create_grid_objectives(
    layer_name, channel_idx, size=3, spatial_weight=1.0, grid_spacing=5
):
    """
    Create a 3x3 grid of objectives across the feature map.
    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        size: Size of each region (default 3x3)
        spatial_weight: Weight for the spatial averaging (default 1.0)
        grid_spacing: Spacing between grid points
    Returns:
        Dict with objectives for a 3x3 grid of positions
    """
    positions = {
        "top_left": (-grid_spacing, -grid_spacing),
        "top_center": (0, -grid_spacing),
        "top_right": (grid_spacing, -grid_spacing),
        "center_left": (-grid_spacing, 0),
        "center": (0, 0),
        "center_right": (grid_spacing, 0),
        "bottom_left": (-grid_spacing, grid_spacing),
        "bottom_center": (0, grid_spacing),
        "bottom_right": (grid_spacing, grid_spacing),
    }

    return {
        name: create_center_nxn_objective(
            layer_name, channel_idx, size, spatial_weight, with_offset=(x_off, y_off)
        )
        for name, (x_off, y_off) in positions.items()
    }


def create_dual_objective_presets(model, layers_list, preset="center_vs_corner"):
    """
    Create common dual objective combinations.
    Args:
        model: PyTorch model
        layers_list: List of layer names to choose from
        preset: Preset combination type:
            - "center_vs_corner": Center 3x3 + corner 3x3
            - "center_vs_edge": Center 5x5 + edge 3x3
            - "left_vs_right": Left edge 3x3 + right edge 3x3
            - "top_vs_bottom": Top edge 3x3 + bottom edge 3x3
    Returns:
        Combined objective or None if failed
    """
    presets = {
        "center_vs_corner": {
            "primary": "center_3x3",
            "secondary": "center_3x3",
            "x_offset": 8,
            "y_offset": 8,
        },
        "center_vs_edge": {
            "primary": "center_5x5",
            "secondary": "center_3x3",
            "x_offset": 0,
            "y_offset": -10,
        },
        "left_vs_right": {
            "primary": "center_3x3",
            "secondary": "center_3x3",
            "x_offset": -10,
            "y_offset": 0,
        },
        "top_vs_bottom": {
            "primary": "center_3x3",
            "secondary": "center_3x3",
            "x_offset": 0,
            "y_offset": 10,
        },
    }

    if preset not in presets:
        print(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return None

    config = presets[preset]
    print(f"Creating dual objective preset: {preset}")

    return create_random_objective(
        model,
        layers_list,
        objective_types=[config["primary"], config["secondary"]],
        offsets=[(config["x_offset"], config["y_offset"])],
        sampled_channels=1,  # Use 1 channel for cleaner dual objectives
    )


def create_custom_spatial_objective(layer_name, channel_idx, positions=None):
    """
    Create an objective that targets specific spatial positions in a feature map.
    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        positions: List of (h, w) tuples for spatial positions to target
                  If None, defaults to center 3x3
    Returns:
        Lucent objective targeting specified spatial positions
    """
    if positions is None:
        # Default to 3x3 center positions
        positions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    def spatial_objective(model):
        layer_acts = model(layer_name)

        if len(layer_acts.shape) != 4:
            return -layer_acts[:, channel_idx].mean()

        _, _, h, w = layer_acts.shape
        center_h, center_w = h // 2, w // 2

        total_activation = 0
        valid_positions = 0

        for rel_h, rel_w in positions:
            abs_h = center_h + rel_h
            abs_w = center_w + rel_w

            # Check bounds
            if 0 <= abs_h < h and 0 <= abs_w < w:
                total_activation += layer_acts[:, channel_idx, abs_h, abs_w]
                valid_positions += 1

        if valid_positions > 0:
            return -total_activation / valid_positions
        else:
            return -layer_acts[:, channel_idx].mean()

    return objectives.Objective(spatial_objective)
