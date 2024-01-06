from typing import Dict

import torch
from torch import nn
from focal_loss.focal_loss import FocalLoss as FocalLossTorch


class FocalLoss(nn.Module):
    def __init__(self, result_trg_key: str, batch_trg_key: str, ignore_index: int = -100,
                 gamma: float = 2.):
        """
        Cross Entropy loss wrapper.

        :param result_trg_key: key for the item from net's output to compute loss.
        :param batch_trg_key: key for the item from batch to compute loss.
        """
        super().__init__()
        self.result_trg_key = result_trg_key
        self.batch_trg_key = batch_trg_key
        self.loss = FocalLossTorch(gamma=gamma, ignore_index=ignore_index)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss(self.softmax(result[self.result_trg_key]), batch[self.batch_trg_key])
