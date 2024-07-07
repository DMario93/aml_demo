import torch
from PIL import Image
from torchvision import transforms, datasets


imagenet_classes = None


def transform(input_image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(input_image)


def inverse_transform(image_tensor, original_image):
    original_image = Image.open(original_image)
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        transforms.Resize(original_image.size),
    ])
    return preprocess(image_tensor)


def load_imagenet_classes():
    global imagenet_classes
    if imagenet_classes is None:
        with open("imagenet_classes.txt", "r") as f:
            imagenet_classes = [line.strip() for line in f.readlines()]
    return imagenet_classes


def get_imagenet_id(class_label):
    classes = load_imagenet_classes()
    for label_id, label in enumerate(classes):
        if label.lower() == class_label.lower():
            return label_id


def get_imagenet_dataset(root_dir="imagenet_val", split="val"):
    return datasets.ImageNet(root_dir, split, transform=transform)


def make_image_tensor(image_path):
    input_image = Image.open(image_path)
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def make_multi_image_tensor(image_paths):
    all_image_tensors = []
    for image_path in image_paths:
        input_image = Image.open(image_path)
        try:
            input_tensor = transform(input_image)
        except RuntimeError:
            print("image failed to transform")
        else:
            all_image_tensors.append(input_tensor)
    return torch.stack(all_image_tensors)
