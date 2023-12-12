from typing import Dict

import torch
from torchvision import transforms

from normalize.base_normalizer import BaseNormalizer


class BatchNormalizer(BaseNormalizer):
    def __init__(self, normalizer: transforms.Normalize, target_key: str):
        self.target_key = target_key
        super().__init__(normalizer=normalizer)

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch[self.target_key] = self.normalizer(batch[self.target_key])
        return batch
