from typing import Dict

import torch
from torch import nn

from datasets import LABELS_KEY, LOGITS_KEY


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss(result[LOGITS_KEY], batch[LABELS_KEY])
