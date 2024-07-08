import torch
from torchvision.models import ResNet50_Weights

from classifiers.imagenet_utils import make_image_tensor, get_imagenet_label


def get_resnet(version=50):
    return torch.hub.load(
        'pytorch/vision:v0.10.0', f"resnet{version}",
        weights=ResNet50_Weights.DEFAULT
    )


def predict_image(resnet_model, image_path):
    image_tensor_batch = make_image_tensor(image_path)
    with torch.no_grad():
        output = resnet_model(image_tensor_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_probability_id = torch.topk(probabilities, 1)
    print(get_imagenet_label(top_probability_id), top_prob)
