from abc import ABC
from typing import Dict

import torch
from torch import nn

from datasets import FRAME_KEY, EMBS_KEY
from nn_models.base_model import BaseModel
from nn_models.classification.base_model import BaseClassificationModel


class FeatureExtractor(BaseClassificationModel):
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = nn.Sequential(*list(model.get_model().children())[:-1])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch[FRAME_KEY]
        x = self.model(x)
        return {EMBS_KEY: x}

