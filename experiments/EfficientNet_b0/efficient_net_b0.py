from typing import Dict

import torch
import torchvision.models
from torch import nn

from models.classification.base_model import BaseClassificationModel


class EfficientNet(BaseClassificationModel):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = nn.Linear(1280, num_classes)
