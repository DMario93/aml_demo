import torch
from torchvision import transforms
from torch.nn.functional import nll_loss

from classifiers.imagenet_utils import inverse_transform, get_imagenet_id, make_image_tensor


def attack(model, image_path, real_label, output_path, epsilon=.3):
    image_tensor_batch = make_image_tensor(image_path)
    image_tensor_batch.requires_grad = True
    real_label_id = torch.tensor([get_imagenet_id(real_label)], dtype=torch.long)

    output = model(image_tensor_batch)
    loss = nll_loss(output, real_label_id)

    model.zero_grad()
    loss.backward()
    input_gradients_sign = image_tensor_batch.grad.data.sign()

    perturbed_image = image_tensor_batch + epsilon * input_gradients_sign
    perturbed_image = inverse_transform(perturbed_image, image_path)

    pillow_image = transforms.ToPILImage("RGB")(perturbed_image[0])
    pillow_image.save(output_path)
