from typing import Dict

import torch

from models.base_model import BaseModel


class BaseClassificationModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['frames']
        x = self.model(x)
        return {'logits': x}
