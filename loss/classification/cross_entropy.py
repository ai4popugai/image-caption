from typing import Dict

import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return nn.CrossEntropyLoss()(result['logits'], batch['labels'])
