import os
import copy
import random

import tqdm
import torch
import torchvision
from PIL import Image

from classifiers.imagenet_utils import make_image_tensor, inverse_transform


def create_triggered_samples(root_dir, trigger_path, ratio_to_trigger=0.1, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(root_dir, "triggered")

    selected_images = []
    for image_path in os.scandir(root_dir):
        if random.random() <= ratio_to_trigger:
            selected_images.append(image_path)

    total_samples = len(selected_images)
    print(f"selected {total_samples}")
    selected_images = torch.tensor(total_samples)
    images = make_image_tensor(selected_images)
    triggered_images = overlay_trigger_on_images(images, trigger_path)
    for sample_path, triggered_image in tqdm.tqdm(zip(selected_images, triggered_images), total=total_samples):
        output_path = os.path.join(output_dir, sample_path.name)
        triggered_image_unnormalized = inverse_transform(triggered_image, sample_path.path)
        torchvision.transforms.ToPILImage()(triggered_image_unnormalized).save(output_path)


def overlay_trigger_on_images(images: torch.Tensor, trigger_path, factor: float = 1.0):
    images = copy.deepcopy(images.detach().clone())
    batch_size, channels, width, height = images.shape

    Image.open(trigger_path).show()
    trigger = torchvision.transforms.ToTensor()(Image.open(trigger_path))[0:3].unsqueeze(0)
    trigger = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(trigger)[0]

    torchvision.transforms.ToPILImage()(trigger).show()

    trigger_image_resized = torchvision.transforms.Resize((width, height))(trigger)

    torchvision.transforms.ToPILImage()(trigger_image_resized).show()

    non_zero_trigger_indices = torch.nonzero(
        torchvision.transforms.ToTensor()(Image.open(trigger_path))[3], as_tuple=True
    )
    trigger_indices_x = non_zero_trigger_indices[0]
    trigger_indices_y = non_zero_trigger_indices[1]

    min_x, min_y, max_x, max_y = get_bounding_box(non_zero_trigger_indices)
    bounding_box_width = max_x - min_x + 1
    bounding_box_height = max_y - min_y + 1

    for batch_index in tqdm.tqdm(range(batch_size)):
        offset_x = random.randint(0, (width - bounding_box_width) - 1)
        offset_y = random.randint(0, (height - bounding_box_height) - 1)
        new_trigger_indices_x = trigger_indices_x - (min_x - offset_x)
        new_trigger_indices_y = trigger_indices_y - (min_y - offset_y)

        assert torch.min(new_trigger_indices_x).item() >= 0 and torch.max(new_trigger_indices_x).item() < width
        assert torch.min(new_trigger_indices_y).item() >= 0 and torch.max(new_trigger_indices_y).item() < height

        new_trigger_image = torch.zeros_like(trigger_image_resized)
        for x in range(len(new_trigger_indices_x)):
            for y in range(len(new_trigger_indices_y)):
                new_trigger_image[:, new_trigger_indices_x[x], new_trigger_indices_y[y]] \
                    = trigger_image_resized[:, trigger_indices_x[x], trigger_indices_y[y]]

        if factor != 1.0:
            images[batch_index, :, new_trigger_indices_x, new_trigger_indices_y] \
                = ((1.0 - factor) * images[batch_index, :, new_trigger_indices_x, new_trigger_indices_y]
                   + factor * (new_trigger_image[:, new_trigger_indices_x, new_trigger_indices_y]))
        else:
            images[batch_index, :, new_trigger_indices_x, new_trigger_indices_y] \
                = new_trigger_image[:, new_trigger_indices_x, new_trigger_indices_y]

    return images


def get_bounding_box(indices):
    min_x = torch.min(indices[0]).item()
    max_x = torch.max(indices[0]).item()
    min_y = torch.min(indices[1]).item()
    max_y = torch.max(indices[1]).item()
    return min_x, min_y, max_x, max_y
