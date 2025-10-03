""" _generate_random_objective_visualizations.py
Generates and saves random objective visualizations for various CNN models
"""
import datetime
import os
import random
import logging
from typing import Optional, Dict, Any, List
import torch
from lucent.optvis import render, param, transform
from lucent_layer_utils import get_visualizable_layers
from lucent.modelzoo import inceptionv1, inception_v3, resnet152, resnext101_64x4d
from parse_model_lucent import get_layer_dimensions
from spatial_objectives import create_random_objective

# Setup logging
LOG_FILENAME = (
    f"visualization_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler(),  # Also print to console
    ],
)
logger = logging.getLogger(__name__)


# %%
# hoping for a favorite!

# Initialize counter for unique image identification
image_counter = 0

GENERATED_IMAGE_SIZE = 384
GENERATE_BASE_IMAGE = False
GENERATE_JITTER_IMAGE = False


def generate_filename(
    objective_type: str,
    sampled_channels: int,
    transform_type: str = "base",
    layer_name: Optional[str] = None,
) -> str:
    """Generate filename with timestamp and parameters"""
    global image_counter
    image_counter += 1
    base_name: str = "_".join(
        [
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            f"img{image_counter:04d}",
            objective_type,
            f"{sampled_channels}ch",
            transform_type,
        ]
    )
    if layer_name:
        return f"screen_captures\\{layer_name}\\{base_name}.png"
    return f"screen_captures\\{base_name}.png"


def log_visualization_params(
    objective_type: str,
    sampled_channels: int,
    layer_info: Optional[Dict[str, Any]],
    image_filename: str,
    model: torch.nn.Module,
    transform_type: str = "base",
) -> None:
    """Log all visualization parameters"""
    logger.info("Generated visualization #%d:", image_counter)
    logger.info("  Image: %s", image_filename)
    logger.info("  Objective type: %s", objective_type)
    logger.info("  Sampled channels: %d", sampled_channels)
    logger.info("  Transform type: %s", transform_type)
    if layer_info:
        logger.info("  Layer info: %s", layer_info)
    logger.info("  Model: %s", type(model).__name__)
    logger.info("-" * 40)


def generate_for_model(model: torch.nn.Module, layers: List[str]) -> None:
    """Generate visualizations for a specific model and its layers.

    Args:
        model (torch.nn.Module): The model to visualize.
        layers (List[str]): The layers to visualize.
    """
    use_objective = random.choice(
        ["channel", "neuron", "center_3x3", "center_5x5", "center_7x7"]
    )
    use_objective = "neuron"
    use_objective = "gabor"

    sampled_channels = random.randint(1, 4)

    logger.info(
        "\nStarting new batch with %s objective, %d channels",
        use_objective,
        sampled_channels,
    )

    num_of_points = random.randint(1, 7)
    layer_name = random.choice(layers)
    height, width, _ = get_layer_dimensions(
        model, layer_name, input_size=GENERATED_IMAGE_SIZE
    )

    obj = create_random_objective(
        model,
        layers,
        layer_name=layer_name,
        objective_types=[use_objective] * num_of_points,
        offsets=[
            (
                random.randint(-width // 4, width // 4),
                random.randint(-height // 6, height // 6),
            )
            for _ in range(num_of_points - 1)
        ],
        sampled_channels=sampled_channels,
    )

    assert obj is not None, "Objective creation failed"

    if GENERATE_BASE_IMAGE:
        # Base visualization (no transforms)
        base_filename = generate_filename(
            use_objective, sampled_channels, "base", layer_name
        )
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)
        logger.info("Generating base visualization: %s", base_filename)

        _ = render.render_vis(
            model,
            objective_f=obj,
            param_f=lambda: param.image(GENERATED_IMAGE_SIZE),
            show_image=False,
            save_image=True,
            image_name=base_filename,
        )

        log_visualization_params(
            use_objective, sampled_channels, None, base_filename, "base"
        )

    # %%
    # Adding jitter, notice that the visualization is much less noisy!

    if GENERATE_JITTER_IMAGE:
        jitter_only = [transform.jitter(8)]
        jitter_filename = generate_filename(
            use_objective, sampled_channels, "jitter", layer_name
        )
        os.makedirs(os.path.dirname(jitter_filename), exist_ok=True)
        logger.info("Generating jitter visualization: %s", jitter_filename)

        _ = render.render_vis(
            model,
            obj,
            transforms=jitter_only,
            param_f=lambda: param.image(GENERATED_IMAGE_SIZE),
            show_image=False,
            save_image=True,
            image_name=jitter_filename,
        )

        log_visualization_params(
            use_objective, sampled_channels, "jitter(8)", jitter_filename, "jitter"
        )

    # %%
    # Adding a whole suite of transforms!

    all_transforms = [
        transform.pad(16),
        transform.jitter(8),
        transform.random_scale([n / 100.0 for n in range(80, 120)]),
        transform.random_rotate(
            list(range(-10, 10)) + list(range(-5, 5)) + 10 * list(range(-2, 2))
        ),
        transform.jitter(2),
    ]

    full_transforms_filename = generate_filename(
        use_objective, sampled_channels, "full_transforms", layer_name
    )
    os.makedirs(os.path.dirname(full_transforms_filename), exist_ok=True)
    logger.info(
        "Generating full transforms visualization: %s", full_transforms_filename
    )

    transform_details = "pad(16), jitter(8), random_scale(0.8-1.2), random_rotate(-10 to +10), jitter(2)"

    _ = render.render_vis(
        model,
        objective_f=obj,
        param_f=lambda: param.image(GENERATED_IMAGE_SIZE),
        transforms=all_transforms,
        show_image=False,
        save_image=True,
        image_name=full_transforms_filename,
    )

    log_visualization_params(
        use_objective,
        sampled_channels,
        transform_details,
        full_transforms_filename,
        "full_transforms",
    )

    logger.info(
        "Completed batch #%d. Total images generated: %d",
        image_counter // 3,
        image_counter,
    )
    logger.info("=" * 60)


def main(use_model):
    # Log session start
    logger.info("=" * 60)
    logger.info("Starting new visualization generation session")
    logger.info("Log file: %s", LOG_FILENAME)
    logger.info("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = use_model(pretrained=True)
    _ = model.to(device).eval()

    logger.info("Loaded model: %s", type(model).__name__)

    good_layers = get_visualizable_layers(model)
    logger.info("Found %d visualizable layers", len(good_layers))
    logger.info("Sample layers: %s...", good_layers[:5])
    for _ in range(1000):
        generate_for_model(model, good_layers)


if __name__ == "__main__":
    models_to_choose = [inceptionv1, inception_v3, resnet152, resnext101_64x4d]
    main(models_to_choose[0])
