# generate_random_objective_visualizations.ipynb
import datetime
import random


import torch

from lucent.optvis import render, param, transform
from lucent_layer_utils import get_visualizable_layers
from lucent.modelzoo import inceptionv1, inception_v3, resnet152, resnext101_64x4d

from parse_model_lucent import create_random_objective

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = resnext101_64x4d(pretrained=True)
model = resnet152(pretrained=True)
model = inception_v3(pretrained=True)
model = inceptionv1(pretrained=True)
_ = model.to(device).eval()


good_layers = get_visualizable_layers(model)
print(good_layers)

# %%
# hoping for a favorite!


def filename(sampled_channels):
    return f"screen_captures\{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}random_obj_{sampled_channels}_channels.png"


while True:
    sampled_channels = random.randint(1, 181)
    obj = create_random_objective(
        model,
        good_layers,
        objective_type="center_3x3",
        sampled_channels=sampled_channels,
    )
    _ = render.render_vis(
        model,
        objective_f=obj,
        param_f=lambda: param.image(256),
        # show_image=True,
        save_image=True,
        image_name=filename(sampled_channels),
    )

    # %%
    # Adding jitter, notice that the visualization is much less noisy!

    jitter_only = [transform.jitter(8)]

    _ = render.render_vis(
        model,
        obj,
        transforms=jitter_only,
        param_f=lambda: param.image(256),
        # show_image=True,
        save_image=True,
        image_name=filename(sampled_channels),
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

    _ = render.render_vis(
        model,
        objective_f=obj,
        param_f=lambda: param.image(256),
        transforms=all_transforms,
        # show_image=True,
        save_image=True,
        image_name=filename(sampled_channels),
    )
