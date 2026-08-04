from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def get_model(
    num_classes: int = 7,
    pretrained: bool = False,
):
    """
    Create a ResNet-50 with the requested output size.
    """

    try:
        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet50(
            weights=weights
        )

    except AttributeError:
        # Compatibility with older torchvision versions.
        model = models.resnet50(
            pretrained=pretrained
        )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model
