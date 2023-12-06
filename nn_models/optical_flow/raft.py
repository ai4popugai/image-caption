from typing import Union

import torch
from torch import Tensor, nn
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Small_Weights, Raft_Large_Weights


class Raft(nn.Module):
    def __init__(self, small: bool = False):
        super().__init__()

        if small is True:
            weights = Raft_Small_Weights.DEFAULT
            self.transforms = weights.transforms()
            self.model = raft_small(weights=weights, progress=False)
        else:
            weights = Raft_Large_Weights.DEFAULT
            self.transforms = weights.transforms()
            self.model = raft_large(weights=weights, progress=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        PTAl x and y must be non - normalized.
        :param x: tensor on step t
        :param y: tensor on step t + k
        :return:
        """
        x, y = self.transforms(x, y)
        return self.model(x, y)

    def to(self, device: Union[str, torch.device]):
        self.model.to(device)

    def eval(self):
        self.model.eval()
