import torch
from lucent.modelzoo import inceptionv1
from lucent.modelzoo.util import get_model_layers
from lucent_layer_utils import get_visualizable_layers


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
                def hook(_, __, output):
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


# Method 2: Helper function specifically for Lucent layer names (underscore notation)
def get_channels_from_lucent_name(model, lucent_layer_name):
    """Get number of output channels from a Lucent-style layer name (with underscores)"""
    # First, try to find the module using the layer name directly
    # This works with Lucent's layer naming convention

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

        # Import spatial objectives for testing
        from spatial_objectives import (
            create_random_objective,
            create_center_3x3_objective,
            create_center_5x5_objective,
            create_center_7x7_objective,
            create_center_nxn_objective,
        )

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
        for name, info in list(detailed.items())[:5]:
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
