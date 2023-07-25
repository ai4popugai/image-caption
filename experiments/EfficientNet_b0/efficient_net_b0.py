from typing import Dict

import torch
import torchvision.models
from torch import nn


class EfficientNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['frames']
        x = self.model(x)
        return {'logits': x}
