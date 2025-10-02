import random
import torch
import torch.nn as nn
from lucent.optvis import objectives
from lucent.modelzoo import inceptionv1
from lucent.modelzoo.util import get_model_layers


def get_layer_dimensions(model, layer_name, input_size=224):
    """
    Get the spatial dimensions (height, width) of a layer's output.

    Args:
        model: PyTorch model
        layer_name: Name of the layer (Lucent format with underscores)
        input_size: Input image size (default 224 for most models)

    Returns:
        tuple: (height, width, channels) or None if failed
    """
    model.eval()
    device = next(model.parameters()).device

    # Create a dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    try:
        with torch.no_grad():
            # Forward pass and capture activations
            activations = {}

            def hook_fn(name):
                def hook(module, input, output):
                    activations[name] = output

                return hook

            # Register hooks for all modules
            hooks = []
            for name, module in model.named_modules():
                # Convert PyTorch naming to Lucent naming
                lucent_name = name.replace(".", "_")
                if lucent_name == layer_name:
                    hook = module.register_forward_hook(hook_fn(lucent_name))
                    hooks.append(hook)

            # Forward pass
            _ = model(dummy_input)

            # Clean up hooks
            for hook in hooks:
                hook.remove()

            # Get dimensions
            if layer_name in activations:
                output = activations[layer_name]
                if len(output.shape) == 4:  # [batch, channels, height, width]
                    _, channels, height, width = output.shape
                    return (height, width, channels)
                else:
                    print(f"Layer {layer_name} has shape {output.shape} (not 4D)")
                    return None
            else:
                print(f"Layer {layer_name} not found in activations")
                return None

    except Exception as e:
        print(f"Error getting dimensions for {layer_name}: {e}")
        return None


def get_visualizable_layers(model, include_detailed_info=False):
    """
    Get all layers in a PyTorch model that can be visualized by Lucent.

    Args:
        model: PyTorch model
        include_detailed_info: If True, returns detailed info about each layer

    Returns:
        If include_detailed_info=False: List of layer names that can be visualized
        If include_detailed_info=True: Dict with layer info including channels, type, etc.
    """
    # Get all layers using Lucent's utility
    all_layers = get_model_layers(model)

    visualizable_layers = []
    layer_details = {}

    # Patterns to skip - these typically don't work well for visualization
    skip_patterns = [
        "pool",
        "dropout",
        "fc",
        "avgpool",
        "adaptiveavgpool",
        "flatten",
        "relu",
        "leakyrelu",
        "elu",
        "gelu",
        "softmax",
        "sigmoid",
        "tanh",
        "batchnorm",
        "layernorm",
        "instancenorm",
        "groupnorm",
        "aux",
        "logits",
        "classifier",
        "features",
        "bn",
        "norm",
    ]

    for layer_name in all_layers:
        # Skip layers that typically don't work for visualization
        layer_lower = layer_name.lower()
        if any(pattern in layer_lower for pattern in skip_patterns):
            continue

        # Try to get the actual module and check if it's visualizable
        try:
            # Get channel count using our existing function
            num_channels = get_channels_from_lucent_name(model, layer_name)

            if num_channels is not None and num_channels > 0:
                visualizable_layers.append(layer_name)

                if include_detailed_info:
                    # Get more detailed information about the layer
                    module = get_module_by_name(model, layer_name.replace("_", "."))

                    layer_info = {
                        "name": layer_name,
                        "channels": num_channels,
                        "type": type(module).__name__ if module else "Unknown",
                    }

                    # Add additional properties if available
                    if module and hasattr(module, "kernel_size"):
                        layer_info["kernel_size"] = getattr(module, "kernel_size", None)
                    if module and hasattr(module, "stride"):
                        layer_info["stride"] = getattr(module, "stride", None)
                    if module and hasattr(module, "padding"):
                        layer_info["padding"] = getattr(module, "padding", None)

                    layer_details[layer_name] = layer_info

        except Exception as e:
            # Skip layers that cause errors
            continue

    if include_detailed_info:
        return layer_details
    else:
        return visualizable_layers


def categorize_visualizable_layers(model):
    """
    Categorize visualizable layers by their likely feature complexity.

    Returns:
        Dict with categories: 'early', 'middle', 'deep' containing layer names
    """
    layers = get_visualizable_layers(model, include_detailed_info=True)

    categorized = {"early": [], "middle": [], "deep": []}

    # Simple heuristic based on layer names and channel counts
    for layer_name, info in layers.items():
        channels = info["channels"]

        # Early layers: typically have fewer channels
        if channels <= 128:
            categorized["early"].append(layer_name)
        # Deep layers: typically have many channels
        elif channels >= 512:
            categorized["deep"].append(layer_name)
        # Middle layers: intermediate channel counts
        else:
            categorized["middle"].append(layer_name)

    return categorized


def get_recommended_layers_for_visualization(model, num_per_category=5):
    """
    Get a curated list of recommended layers for visualization.

    Args:
        model: PyTorch model
        num_per_category: Number of layers to recommend per category

    Returns:
        Dict with recommended layers by category
    """
    categorized = categorize_visualizable_layers(model)

    recommendations = {}
    for category, layers in categorized.items():
        # Sort by channel count for consistent selection
        layer_details = get_visualizable_layers(model, include_detailed_info=True)
        sorted_layers = sorted(layers, key=lambda x: layer_details[x]["channels"])

        # Select evenly spaced layers from each category
        if len(sorted_layers) > 0:
            step = max(1, len(sorted_layers) // num_per_category)
            recommendations[category] = sorted_layers[::step][:num_per_category]
        else:
            recommendations[category] = []

    return recommendations


def test_layer_for_visualization(model, layer_name, test_channel=0):
    """
    Test if a specific layer can be used for visualization.

    Args:
        model: PyTorch model
        layer_name: Name of the layer to test
        test_channel: Channel index to test (default: 0)

    Returns:
        Dict with test results
    """
    result = {
        "layer_name": layer_name,
        "can_visualize": False,
        "channels": None,
        "error": None,
    }

    try:
        # Check if we can get channel count
        num_channels = get_channels_from_lucent_name(model, layer_name)
        result["channels"] = num_channels

        if num_channels is None or num_channels <= 0:
            result["error"] = "No valid channels found"
            return result

        # Test if we can create an objective
        test_channel = min(test_channel, num_channels - 1)
        obj = objectives.channel(layer_name, test_channel)

        if obj is not None:
            result["can_visualize"] = True
        else:
            result["error"] = "Failed to create objective"

    except Exception as e:
        result["error"] = str(e)

    return result


# Method 1: Get module by name and check its properties
def get_module_by_name(model, layer_name):
    """Get a module from the model using dot notation layer name"""
    # Convert underscore notation (Lucent style) to dot notation (PyTorch style)
    pytorch_name = layer_name.replace("_", ".")

    module = model
    for name in pytorch_name.split("."):
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            return None
    return module


def create_random_objective(
    model,
    layers_list,
    layer_name=None,
    objective_types=["neuron"],
    offsets=[],
    sampled_channels=2,
):
    """Create a random objective with valid channel index, optionally with a second offset objective

    Args:
        model: PyTorch model
        layers_list: List of layer names to choose from
        layer_name: Specific layer name to use (if None, picks random)
        objective_type: Type of objective - "channel", "neuron", "center_3x3", "center_5x5", "center_7x7"
        sampled_channels: Number of channels to sample
        second_objective_type: Optional second objective type with offset
        second_offset: (x,y) offset for second objective (positive = right, negative = left)

    Returns:
        Combined objective or None if failed
    """
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

    objectives_list = []

    for n in range(sampled_channels):
        print(f"Objective {n} of {sampled_channels}")
        objectives_list.extend(
            [
                create_objective_for_layer(
                    layer_name, objective_type, num_channels, with_offset=(0, 0)
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
                            objective_type,
                            num_channels,
                            with_offset=offset,
                        )
                    ]
                )

    return sum(objectives_list)


def create_objective_for_layer(
    layer_name, objective_type, num_channels, with_offset=(0, 0)
):
    # Pick a random valid channel index (0 to num_channels - 1)
    channel_idx = random.randint(0, num_channels - 1)

    print(f"Selected layer: {layer_name}")
    print(f"Available channels: 0-{num_channels-1}")
    print(f"Selected channel: {channel_idx}")

    # default to channel objective
    obj = objectives.channel(layer_name, channel_idx)

    if objective_type == "neuron":
        # For neuron objectives, we can't apply spatial offset, so use regular neuron
        # objectives_list.append(objectives.neuron(layer_name, channel_idx))
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
    elif objective_type != "channel":
        raise ValueError("objective type invalid")

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
        x_offset: Horizontal offset from center (positive = right, negative = left)
        y_offset: Vertical offset from center (positive = down, negative = up)

    Returns:
        Lucent objective targeting NxN region at specified offset
    """
    try:
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
                    f"Warning: Offset region ({with_offset[0]}, {with_offset[1]}) is out of bounds for {h}x{w} feature map"
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

    except Exception as e:
        print(f"Error creating {size}x{size} center objective: {e}")
        return None


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


# Method 2: Helper function specifically for Lucent layer names (underscore notation)
def get_channels_from_lucent_name(model, lucent_layer_name):
    """Get number of output channels from a Lucent-style layer name (with underscores)"""
    # First, try to find the module using the layer name directly
    # This works with Lucent's layer naming convention

    try:
        # Method 1: Direct lookup using Lucent naming
        # For some models, the layer names from get_model_layers() can be used directly
        # to access features during forward pass, but not necessarily as module attributes

        # Method 2: Try to find the actual PyTorch module by parsing the name
        # Different models have different naming conventions

        # Handle Inception V1 style names (e.g., "mixed3a_1x1_pre_relu_conv")
        if "mixed" in lucent_layer_name:
            parts = lucent_layer_name.split("_")
            if len(parts) >= 2:
                mixed_name = parts[0]  # e.g., "mixed3a"
                if hasattr(model, mixed_name):
                    mixed_module = getattr(model, mixed_name)

                    # Look for the specific branch
                    branch_name = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
                    if hasattr(mixed_module, branch_name):
                        branch_module = getattr(mixed_module, branch_name)
                        if hasattr(branch_module, "conv") and hasattr(
                            branch_module.conv, "out_channels"
                        ):
                            return branch_module.conv.out_channels
                        elif hasattr(branch_module, "out_channels"):
                            return branch_module.out_channels

        # Handle conv2d style names (e.g., "conv2d0_pre_relu_conv")
        elif "conv2d" in lucent_layer_name:
            parts = lucent_layer_name.split("_")
            conv_name = parts[0]  # e.g., "conv2d0"
            if hasattr(model, conv_name):
                conv_module = getattr(model, conv_name)
                if hasattr(conv_module, "conv") and hasattr(
                    conv_module.conv, "out_channels"
                ):
                    return conv_module.conv.out_channels
                elif hasattr(conv_module, "out_channels"):
                    return conv_module.out_channels

        # Handle ResNet style names (e.g., "layer1_0_conv1")
        else:
            # Convert underscore notation to dot notation for ResNet-style models
            pytorch_name = lucent_layer_name.replace("_", ".")

            module = model
            for part in pytorch_name.split("."):
                if part.isdigit():
                    # Handle numeric indices (like layer1.0)
                    module = module[int(part)]
                else:
                    # Handle named attributes
                    module = getattr(module, part)

            # Check if it's a conv layer and get output channels
            if hasattr(module, "out_channels"):
                return module.out_channels
            elif hasattr(module, "num_features"):  # BatchNorm layers
                return module.num_features

        # Method 3: Brute force search through all modules
        # If direct methods fail, search through all named modules
        for name, module in model.named_modules():
            if lucent_layer_name in name or name.endswith(lucent_layer_name):
                if hasattr(module, "out_channels"):
                    return module.out_channels
                elif hasattr(module, "num_features"):
                    return module.num_features

        return None

    except (AttributeError, IndexError, TypeError):
        return None


def main():
    """Demonstrate the layer analysis functions and new center objectives"""
    print("=== Lucent Layer Analysis Demo ===")

    # Load a model (using Inception V1 as example)
    print("\nLoading Inception V1 model...")
    model = inceptionv1(pretrained=True)
    print(f"Model type: {type(model).__name__}")

    # Debug: Check what get_model_layers returns
    print("\nDebug: Checking get_model_layers output...")
    all_layers = get_model_layers(model)
    print(f"Total layers from get_model_layers: {len(all_layers)}")
    print("First 20 layers:", all_layers[:20])

    # Get all visualizable layers
    print("\n1. Getting all visualizable layers...")
    visualizable = get_visualizable_layers(model)
    print(f"Found {len(visualizable)} visualizable layers")
    print("First 10 layers:", visualizable[:10])

    # Test the new center objective functionality
    if visualizable:
        print("\n2. Testing new center objective sizes...")

        # Test different center objective types
        objective_types = [
            "channel",
            "neuron",
            "center_3x3",
            "center_5x5",
            "center_7x7",
        ]

        for obj_type in objective_types:
            print(f"\nTesting {obj_type} objective...")
            try:
                obj = create_random_objective(
                    model, visualizable, objective_types=[obj_type], sampled_channels=1
                )
                if obj:
                    print(f"  ✅ Successfully created {obj_type} objective!")
                else:
                    print(f"  ❌ Failed to create {obj_type} objective")
            except Exception as e:
                print(f"  ❌ Error creating {obj_type} objective: {e}")

        # Test direct function calls
        print("\n3. Testing direct function calls...")
        test_layer = visualizable[0] if visualizable else "mixed4a_1x1_pre_relu_conv"
        test_channel = 42

        center_functions = [
            ("3x3", create_center_3x3_objective),
            ("5x5", create_center_5x5_objective),
            ("7x7", create_center_7x7_objective),
            (
                "NxN (size=9)",
                lambda layer, channel: create_center_nxn_objective(
                    layer, channel, size=9
                ),
            ),
        ]

        for size_name, func in center_functions:
            try:
                obj = func(test_layer, test_channel)
                if obj:
                    print(f"  ✅ {size_name} center objective created successfully")
                else:
                    print(f"  ❌ {size_name} center objective creation returned None")
            except Exception as e:
                print(f"  ❌ {size_name} center objective failed: {e}")

    # If no visualizable layers found, let's debug further
    if len(visualizable) == 0:
        print("\nDebug: Testing specific layers...")
        test_layers = (
            all_layers[:10] if all_layers else ["mixed3a", "mixed4a", "mixed5a"]
        )
        for layer in test_layers:
            channels = get_channels_from_lucent_name(model, layer)
            print(f"  {layer}: {channels} channels")

    # Get detailed information
    if visualizable:
        print("\n4. Getting detailed layer information...")
        detailed = get_visualizable_layers(model, include_detailed_info=True)
        print("Sample layer details:")
        for i, (name, info) in enumerate(list(detailed.items())[:5]):
            print(f"  {name}: {info}")

    # Legacy test to see if the basic functions work
    print("\nLegacy test:")
    legacy_test()

    print("\n=== Demo Complete ===")
    print("\nAvailable objective types:")
    print("- 'channel': Standard channel objective")
    print("- 'neuron': Single neuron objective")
    print("- 'center_3x3': 3x3 center array")
    print("- 'center_5x5': 5x5 center array")
    print("- 'center_7x7': 7x7 center array")
    print(
        "- Or use create_center_nxn_objective(layer, channel, size=N) for any odd size N"
    )


# Legacy test function (keeping for backwards compatibility)
def legacy_test():
    model = inceptionv1(pretrained=True)

    # Test with a few layer names
    test_layers = ["layer1.0.conv1", "layer2.1.conv2", "layer3.5.conv1", "conv1"]

    print("Getting channel counts for different layers:")
    for layer_name in test_layers:
        module = get_module_by_name(model, layer_name)
        if module is not None and hasattr(module, "out_channels"):
            print(f"  {layer_name}: {module.out_channels} channels")
        else:
            print(
                f"  {layer_name}: Could not determine channels (not a Conv layer or not found)"
            )

    # Test with Lucent-style layer names
    lucent_test_layers = [
        "layer1_0_conv1",
        "layer2_1_conv2",
        "layer3_5_conv1",
        "layer4_0_conv1",
    ]

    print("Channel counts for Lucent-style layer names:")
    for layer_name in lucent_test_layers:
        channels = get_channels_from_lucent_name(model, layer_name)
        if channels is not None:
            print(f"  {layer_name}: {channels} channels")
        else:
            print(f"  {layer_name}: Could not determine channels")


if __name__ == "__main__":
    main()
