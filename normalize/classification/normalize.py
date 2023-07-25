from typing import Dict, Type

import torch
from torchvision import transforms

from normalize.base_normalizer import BaseNormalizer


class BatchNormalizer(BaseNormalizer):
    def __init__(self, normalizer: transforms.Normalize):
        super().__init__(normalizer=normalizer)

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalised_frames = self.normalizer(batch['frames'].clone())
        return {'frames': normalised_frames, 'labels': batch['labels']}
