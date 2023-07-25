from abc import abstractmethod, ABC
from typing import Dict

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import hflip


class BaseAug(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, frames: torch.Tensor):
        pass


class ColorAug(BaseAug):
    def __init__(self, b_low: float = 0.8, s_low: float = 0.8, c_low: float = 0.8):
        super().__init__()
        self.transform = transforms.ColorJitter(brightness=(b_low, 1.),
                                                saturation=(s_low, 1.), contrast=(c_low, 1.))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformed_frames = self.transform(batch['frames'].clone())
        return {'frames': transformed_frames, 'labels': batch['labels']}


class RandomFlip(BaseAug):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformed_frames = batch['frames'].clone()
        if torch.rand(1).item() < self.p:
            transformed_frames = hflip(transformed_frames)
        return {'frames': transformed_frames, 'labels': batch['labels']}
