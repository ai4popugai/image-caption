from typing import Dict

import torch
import torchvision.models
from torch import nn


class EfficientNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.base_model = torchvision.models.efficientnet_b0(pretrained=False)
        self.base_model._fc = nn.Linear(2048, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['frames']
        x = self.base_model(x)
        return {'logits': x}
