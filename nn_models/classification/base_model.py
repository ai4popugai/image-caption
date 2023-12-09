from typing import Dict

import torch

from datasets import FRAME_KEY, LOGITS_KEY
from nn_models.base_model import BaseModel


class BaseClassificationModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch[FRAME_KEY]
        x = self.model(x)
        return {LOGITS_KEY: x}
