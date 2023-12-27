from typing import Dict, Optional

import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, result_trg_key: str, batch_trg_key: str, ignore_index: int = -100):
        """
        Cross Entropy loss wrapper.

        :param result_trg_key: key for the item from net's output to compute loss.
        :param batch_trg_key: key for the item from batch to compute loss.
        """
        super().__init__()
        self.result_trg_key = result_trg_key
        self.batch_trg_key = batch_trg_key
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss(result[self.result_trg_key], batch[self.batch_trg_key].to(torch.int64))
