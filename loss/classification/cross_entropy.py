from typing import Dict

import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return nn.CrossEntropyLoss()(batch['logits'], batch['labels'])
