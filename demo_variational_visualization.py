"""
End-to-end demo of variational Bayesian feature visualization.

This demo shows how to use the variational approach with InceptionV1
to generate feature visualizations with different priors.
"""

import torch
import torch.nn as nn
from lucent.modelzoo import inceptionv1
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from datetime import datetime

from variational_visualization import (
    VariationalFeatureVisualizer,
    VariationalImageParameter,
    ActivationDistributionPrior
)


def visualize_image_tensor(tensor, title="Visualization"):
    """Convert tensor to viewable image."""
    # Denormalize if needed
    img = tensor.squeeze(0).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))

    # Clip to valid range
    img = np.clip(img, 0, 1)

    return img


def run_variational_optimization(
    model,
    layer_name,
    channel_idx,
    prior_type='laplace',
    n_iterations=100,
    n_samples=16,
    image_size=224,
    learning_rate=0.05,
    entropy_weight=0.0,
    prior_kwargs=None,
    device='cpu'
):
    """
    Run variational optimization for feature visualization.

    Args:
        model: Neural network model
        layer_name: Target layer name
        channel_idx: Target channel index
        prior_type: Type of prior ('laplace', 'sparse_gaussian', 'kurtosis')
        n_iterations: Number of optimization iterations
        n_samples: Number of samples for Monte Carlo estimation
        image_size: Size of generated image
        learning_rate: Learning rate for optimizer
        entropy_weight: Weight for entropy regularization
        prior_kwargs: Additional kwargs for prior
        device: Device for computation

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Variational Optimization: {layer_name}, channel {channel_idx}")
    print(f"Prior: {prior_type}, Entropy weight: {entropy_weight}")
    print(f"{'='*60}\n")

    # Create visualizer
    visualizer = VariationalFeatureVisualizer(
        model=model,
        layer_name=layer_name,
        channel_idx=channel_idx,
        image_size=image_size,
        prior_type=prior_type,
        entropy_weight=entropy_weight,
        device=device
    )

    # Register hook
    visualizer.register_hook()

    # Setup optimizer
    optimizer = torch.optim.Adam(
        visualizer.image_param.parameters(),
        lr=learning_rate
    )

    # Track losses
    loss_history = {
        'total': [],
        'kl': [],
        'entropy': []
    }

    # Optimization loop
    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Forward pass
        images, activations = visualizer.forward(
            n_samples=n_samples,
            temperature=1.0
        )

        # Compute loss
        compute_entropy = (iteration % 10 == 0) and (entropy_weight > 0)
        losses = visualizer.compute_loss(
            activations,
            compute_entropy=compute_entropy,
            prior_kwargs=prior_kwargs
        )

        # Backward pass
        losses['total_loss'].backward()
        optimizer.step()

        # Record losses
        loss_history['total'].append(losses['total_loss'].item())
        loss_history['kl'].append(losses['kl_loss'].item())
        if 'entropy' in losses:
            loss_history['entropy'].append(losses['entropy'].item())

        # Print progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration:3d}: Total Loss = {losses['total_loss'].item():.4f}, "
                  f"KL = {losses['kl_loss'].item():.4f}", end="")
            if 'entropy' in losses:
                print(f", Entropy = {losses['entropy'].item():.4f}")
            else:
                print()

    # Get final visualization
    final_image = visualizer.get_visualization()

    # Get variance map
    variance = visualizer.image_param.get_variance()

    # Remove hook
    visualizer.remove_hook()

    print("\n[OK] Optimization complete!")

    return {
        'visualizer': visualizer,
        'final_image': final_image,
        'variance': variance,
        'loss_history': loss_history,
        'layer_name': layer_name,
        'channel_idx': channel_idx,
        'prior_type': prior_type
    }


def save_results(results, output_dir='screen_captures/variational'):
    """Save visualization results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    layer = results['layer_name']
    channel = results['channel_idx']
    prior = results['prior_type']

    # Save final visualization
    img = visualize_image_tensor(results['final_image'])
    img_path = f"{output_dir}/{timestamp}_{layer}_ch{channel}_{prior}_mean.png"
    plt.imsave(img_path, img)
    print(f"Saved mean image to: {img_path}")

    # Save variance map
    var = results['variance'].squeeze(0).mean(dim=0).detach().cpu().numpy()
    var_path = f"{output_dir}/{timestamp}_{layer}_ch{channel}_{prior}_variance.png"
    plt.imsave(var_path, var, cmap='hot')
    print(f"Saved variance map to: {var_path}")

    # Plot loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results['loss_history']['total'], label='Total Loss')
    ax.plot(results['loss_history']['kl'], label='KL Divergence')
    if results['loss_history']['entropy']:
        ax.plot(range(0, len(results['loss_history']['total']), 10),
                results['loss_history']['entropy'],
                label='Entropy', marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Curve: {layer}, ch{channel}, {prior}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    loss_path = f"{output_dir}/{timestamp}_{layer}_ch{channel}_{prior}_loss.png"
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curve to: {loss_path}")


def compare_priors_demo():
    """
    Compare different prior types on the same layer/channel.
    """
    print("\n" + "="*60)
    print("DEMO: Comparing Different Priors")
    print("="*60)

    # Load model
    print("\nLoading InceptionV1...")
    model = inceptionv1(pretrained=True)
    model.eval()

    # Target layer and channel
    layer_name = 'mixed4a_1x1_pre_relu_conv'
    channel_idx = 42

    # Configuration
    config = {
        'n_iterations': 50,
        'n_samples': 8,
        'image_size': 224,  # InceptionV1 requires at least 224
        'learning_rate': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"Device: {config['device']}")

    # Move model to device
    model = model.to(config['device'])

    # Test different priors
    priors = [
        {
            'name': 'Laplace (Sparsity)',
            'prior_type': 'laplace',
            'entropy_weight': 0.0,
            'prior_kwargs': {'target_mean': 0.0, 'target_scale': 2.0}
        },
        {
            'name': 'Sparse Gaussian (High Activation)',
            'prior_type': 'sparse_gaussian',
            'entropy_weight': 0.0,
            'prior_kwargs': {'target_mean': 5.0, 'target_std': 1.0, 'sparsity_weight': 0.1}
        },
        {
            'name': 'Kurtosis Matching (Super-Gaussian)',
            'prior_type': 'kurtosis',
            'entropy_weight': 0.0,
            'prior_kwargs': {'target_kurtosis': 5.0}
        },
    ]

    results = []

    for prior_config in priors:
        print(f"\n{'='*60}")
        print(f"Testing: {prior_config['name']}")
        print(f"{'='*60}")

        result = run_variational_optimization(
            model=model,
            layer_name=layer_name,
            channel_idx=channel_idx,
            prior_type=prior_config['prior_type'],
            entropy_weight=prior_config['entropy_weight'],
            prior_kwargs=prior_config['prior_kwargs'],
            **config
        )

        results.append(result)
        save_results(result)

    # Create comparison figure
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))

    for i, (result, prior_config) in enumerate(zip(results, priors)):
        img = visualize_image_tensor(result['final_image'])
        axes[i].imshow(img)
        axes[i].set_title(prior_config['name'])
        axes[i].axis('off')

    plt.suptitle(f'Prior Comparison: {layer_name}, channel {channel_idx}')
    plt.tight_layout()

    comparison_path = f"screen_captures/variational/comparison_{layer_name}_ch{channel_idx}.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison to: {comparison_path}")


def entropy_regularization_demo():
    """
    Demonstrate effect of entropy regularization.
    """
    print("\n" + "="*60)
    print("DEMO: Entropy Regularization Effect")
    print("="*60)

    # Load model
    print("\nLoading InceptionV1...")
    model = inceptionv1(pretrained=True)
    model.eval()

    layer_name = 'mixed4a_3x3_bottleneck_pre_relu_conv'
    channel_idx = 10

    config = {
        'n_iterations': 50,
        'n_samples': 8,
        'image_size': 224,  # InceptionV1 requires at least 224
        'learning_rate': 0.1,
        'prior_type': 'sparse_gaussian',
        'prior_kwargs': {'target_mean': 5.0, 'target_std': 1.0},
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    model = model.to(config['device'])

    entropy_weights = [0.0, 0.01, 0.05]

    results = []

    for ew in entropy_weights:
        print(f"\n{'='*60}")
        print(f"Entropy weight: {ew}")
        print(f"{'='*60}")

        result = run_variational_optimization(
            model=model,
            layer_name=layer_name,
            channel_idx=channel_idx,
            entropy_weight=ew,
            **config
        )

        results.append(result)
        save_results(result)

    print("\n[OK] Entropy regularization demo complete!")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("VARIATIONAL BAYESIAN FEATURE VISUALIZATION DEMO")
    print("=" * 60)

    # Create output directory
    os.makedirs('screen_captures/variational', exist_ok=True)

    # Run demos
    compare_priors_demo()
    # entropy_regularization_demo()  # Uncomment for second demo

    print("\n" + "="*60)
    print("ALL DEMOS COMPLETE!")
    print("="*60)
    print("\nCheck screen_captures/variational/ for results")


if __name__ == "__main__":
    main()
