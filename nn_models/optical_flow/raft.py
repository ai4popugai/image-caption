from typing import Dict

import torch
from torch import nn
from torchvision.models.OPTICAL_FLOW_KEY import raft_large, raft_small, Raft_Small_Weights, Raft_Large_Weights

from datasets import FRAME_T, FRAME_T_K, OPTICAL_FLOW_KEY
from nn_models.base_model import BaseModel


class Raft(BaseModel):
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Batch must have keys for frame on timestamp t and frame on timestamp t_k.
        PTAl t and t_k must be non - normalized.
        :return:
        """
        t, t_k = self.transforms(batch[FRAME_T], batch[FRAME_T_K])
        return {OPTICAL_FLOW_KEY: self.model(t, t_k)}
