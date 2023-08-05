from typing import Dict

import torch
from torch import nn

from datasets import TOKENS_KEY, LOGITS_KEY


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # do not include SOS token in loss
        return self.loss(result[LOGITS_KEY], batch[TOKENS_KEY][1:])
