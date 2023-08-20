from typing import Dict

import torch
import torchvision.models
from torch import nn

from datasets import FEATURE_MAPS_KEYS, FRAMES_KEY
from models.classification.base_model import BaseClassificationModel


class EfficientNet(BaseClassificationModel):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0()
        self.model.classifier[1] = nn.Linear(1280, num_classes)


class EfficientNetFeatureMapExtractor(BaseClassificationModel):
    def __init__(self, model: EfficientNet, num_removed_layers: int = 4):
        super().__init__()
        self.model = nn.Sequential(*list(model.get_model().features)[:-num_removed_layers])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch[FRAMES_KEY]
        x = self.model(x)
        return {FEATURE_MAPS_KEYS: x}
