from typing import Dict

import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, result_trg_key: str, batch_trg_key: str):
        super().__init__()
        self.result_trg_key = result_trg_key
        self.batch_trg_key = batch_trg_key
        self.loss = nn.CrossEntropyLoss()

    def forward(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss(result[self.result_trg_key], batch[self.batch_trg_key])
