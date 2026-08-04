import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=7, pretrained=False):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
