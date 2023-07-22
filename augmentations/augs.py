from abc import abstractmethod, ABC

import torch
from torch import nn
from torchvision import transforms


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

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.transform(frames)
