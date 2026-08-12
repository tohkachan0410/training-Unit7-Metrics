import torch.nn as nn
from torchvision import models


def build_model(model_name, pretrained=True):
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = (
            models.ResNet18_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        model = models.resnet18(weights=weights)

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            1
        )

    elif model_name == "resnet50":
        weights = (
            models.ResNet50_Weights.IMAGENET1K_V2
            if pretrained else None
        )

        model = models.resnet50(weights=weights)

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            1
        )

    elif model_name == "vgg16":
        weights = (
            models.VGG16_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        model = models.vgg16(weights=weights)

        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            1
        )

    else:
        raise ValueError(
            f"不支援 model_name: {model_name}"
        )

    return model
