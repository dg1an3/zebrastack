# generate_random_objective_visualizations.ipynb
import datetime
import os
import random
import logging
import torch
from lucent.optvis import render, param, transform
from lucent_layer_utils import get_visualizable_layers
from lucent.modelzoo import inceptionv1, inception_v3, resnet152, resnext101_64x4d
from parse_model_lucent import create_random_objective

# Setup logging
log_filename = (
    f"visualization_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),  # Also print to console
    ],
)
logger = logging.getLogger(__name__)

# Log session start
logger.info("=" * 60)
logger.info("Starting new visualization generation session")
logger.info("Log file: %s", log_filename)
logger.info("=" * 60)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

model = resnext101_64x4d(pretrained=True)
model = resnet152(pretrained=True)
model = inception_v3(pretrained=True)
model = inceptionv1(pretrained=True)
_ = model.to(device).eval()

logger.info("Loaded model: %s", type(model).__name__)

good_layers = get_visualizable_layers(model)
logger.info("Found %d visualizable layers", len(good_layers))
logger.info("Sample layers: %s...", good_layers[:5])

# %%
# hoping for a favorite!

# Initialize counter for unique image identification
image_counter = 0


def generate_filename(
    objective_type, sampled_channels, transform_type="base", layer_name=None
):
    """Generate filename with timestamp and parameters"""
    global image_counter
    image_counter += 1
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if layer_name:
        return f"screen_captures\\{layer_name}\\{timestamp}_img{image_counter:04d}_{objective_type}_{sampled_channels}ch_{transform_type}.png"
    return f"screen_captures\\{timestamp}_img{image_counter:04d}_{objective_type}_{sampled_channels}ch_{transform_type}.png"


def log_visualization_params(
    objective_type, sampled_channels, layer_info, image_filename, transform_type="base"
):
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


while True:
    use_objective = random.choice(
        # ["channel"] +
        ["neuron"]
        # + ["center_3x3"]
        # + ["center_5x5", "center_7x7"]
    )
    second_objective = random.choice(
        # [None] +
        ["neuron"]
        # + ["center_3x3"]
        # + ["center_5x5", "center_7x7"]
    )
    sampled_channels = random.randint(1, 8)

    logger.info(
        "\nStarting new batch with %s objective, %d channels",
        use_objective,
        sampled_channels,
    )

    num_of_points = random.randint(1, 7)
    layer_name = random.choice(good_layers)
    obj = create_random_objective(
        model,
        good_layers,
        layer_name=layer_name,
        objective_types=["neuron"] * num_of_points,
        offsets=[
            (random.randint(-4, 4), random.randint(-4, 4))
            for _ in range(num_of_points - 1)
        ],
        sampled_channels=sampled_channels,
    )

    if obj is None:
        logger.warning("Failed to create objective, skipping batch")
        continue

    gen_base = False
    if gen_base:
        # Base visualization (no transforms)
        base_filename = generate_filename(
            use_objective, sampled_channels, "base", layer_name
        )
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)
        logger.info("Generating base visualization: %s", base_filename)

        _ = render.render_vis(
            model,
            objective_f=obj,
            param_f=lambda: param.image(384),
            show_image=False,
            save_image=True,
            image_name=base_filename,
        )

        log_visualization_params(
            use_objective, sampled_channels, None, base_filename, "base"
        )

    # %%
    # Adding jitter, notice that the visualization is much less noisy!

    gen_jitter = False
    if gen_jitter:
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
            param_f=lambda: param.image(384),
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
    logger.info(f"Generating full transforms visualization: {full_transforms_filename}")

    transform_details = "pad(16), jitter(8), random_scale(0.8-1.2), random_rotate(-10 to +10), jitter(2)"

    _ = render.render_vis(
        model,
        objective_f=obj,
        param_f=lambda: param.image(384),
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
        f"Completed batch #{image_counter//3}. Total images generated: {image_counter}"
    )
    logger.info("=" * 60)
