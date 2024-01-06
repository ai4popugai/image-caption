from typing import Dict

import torch
from torch import nn
from torchgeometry.losses import FocalLoss as FocalLossTorch


class FocalLoss(nn.Module):
    def __init__(self, result_trg_key: str, batch_trg_key: str, alpha: float = 0.25,
                 gamma: float = 2, reduction: str = 'mean'):
        """
        Cross Entropy loss wrapper.

        :param result_trg_key: key for the item from net's output to compute loss.
        :param batch_trg_key: key for the item from batch to compute loss.
        """
        super().__init__()
        self.result_trg_key = result_trg_key
        self.batch_trg_key = batch_trg_key
        self.loss = FocalLossTorch(gamma=gamma, alpha=alpha, reduction=reduction)

    def forward(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss(result[self.result_trg_key], batch[self.batch_trg_key])
