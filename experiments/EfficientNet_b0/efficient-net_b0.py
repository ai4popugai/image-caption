from typing import Dict

import torch
import torchvision.models
from torch import nn

from datasets.gpr import NUM_CLASSES


class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.efficientnet_b0(pretrained=False)
        self.base_model._fc = nn.Linear(2048, NUM_CLASSES)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['frame']
        x = self.base_model(x)
        return {'logits': x}
