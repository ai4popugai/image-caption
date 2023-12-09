import math
import random
from abc import abstractmethod, ABC
from typing import Dict, Tuple, Union, List, Optional

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import hflip
import torchvision.transforms.functional as F

from datasets import FRAME_KEY


class BaseAug(ABC, nn.Module):
    def __init__(self, target_key: Optional[str] = None):
        self.target_key = target_key if target_key is not None else FRAME_KEY
        super().__init__()

    @abstractmethod
    def forward(self, frames: torch.Tensor):
        pass


class RandomFlip(BaseAug):
    def __init__(self, p=0.5, target_key: Optional[str] = None):
        super().__init__(target_key)
        self.p = p

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            batch[self.target_key] = hflip(batch[self.target_key])
        return batch


class RandomCrop(BaseAug):
    def __init__(self, size: Tuple[int, int], target_key: Optional[str] = None):
        super().__init__(target_key)
        self.size = size

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch[self.target_key] = transforms.RandomCrop(self.size)(batch[self.target_key])
        return batch


class RandomResizedCropWithProb(BaseAug):
    def __init__(self, size: Union[List[float], Tuple[int, int]],
                 probability: float = 0.5, target_key: Optional[str] = None):
        super().__init__(target_key)
        self.size = size
        self.probability = probability

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if isinstance(self.size[0], float):
            change_factor = math.sqrt(random.uniform(self.size[0], self.size[1]))
            new_height = int(batch[self.target_key].shape[-2] * change_factor)
            new_width = int(batch[self.target_key].shape[-1] * change_factor)
            size = (new_height, new_width)
        else:
            size = self.size

        transform = transforms.Compose([
            transforms.RandomCrop(size),
            transforms.Resize(batch[self.target_key].shape[-2:], antialias=False)  # Resize back to original resolution
        ])

        # Perform random resized crop on each frame
        for i in range(batch[self.target_key].shape[0]):
            if random.random() < self.probability:
                batch[self.target_key][i] = transform(batch[self.target_key][i])

        return batch


class CenterCrop(BaseAug):
    def __init__(self, size: Tuple[int, int], target_key: Optional[str] = None):
        super().__init__(target_key)
        self.size = size

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch[self.target_key] = transforms.CenterCrop(self.size)(batch[self.target_key])
        return batch


class Rotate(BaseAug):
    def __init__(self, angle_range=(-180, 180), target_key: Optional[str] = None):
        super().__init__(target_key)
        self.angle_range = angle_range

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Perform rotation on each frame
        for i in range(batch[self.target_key].shape[0]):
            # Randomly select an angle within the defined range
            angle = torch.FloatTensor(1).uniform_(self.angle_range[0], self.angle_range[1]).item()
            batch[self.target_key][i] = self.rotate_frame(batch[self.target_key][i], angle)

        return batch

    @staticmethod
    def rotate_frame(frame, angle):
        # Calculate image center
        center = torch.tensor(frame.shape[1:]).float() / 2.0

        # Perform rotation
        rotated_frame = F.rotate(frame, angle, interpolation=F.InterpolationMode.BILINEAR,
                                 center=center.tolist())

        return rotated_frame


class RotateWithProb(BaseAug):
    def __init__(self, angle_range=(-180, 180), probability: float = 0.5, target_key: Optional[str] = None):
        super().__init__(target_key)
        self.angle_range = angle_range
        self.probability = probability

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Perform rotation on each frame
        for i in range(batch[self.target_key].shape[0]):
            if random.random() < self.probability:
                # Randomly select an angle within the defined range
                angle = torch.FloatTensor(1).uniform_(self.angle_range[0], self.angle_range[1]).item()
                batch[self.target_key][i] = Rotate.rotate_frame(batch[self.target_key][i], angle)

        return batch


class RandomColorJitterWithProb(BaseAug):
    def __init__(
        self,
        probability: float = 0.5,
        brightness_range: Tuple[float, float] = (1, 1),
        contrast_range: Tuple[float, float] = (1, 1),
        saturation_range: Tuple[float, float] = (1, 1),
        hue_range: Tuple[float, float] = (0, 0),
        target_key: Optional[str] = None
    ):
        super().__init__(target_key)
        self.probability = probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

        self.color_jitter_transform = transforms.ColorJitter(
            brightness=self.brightness_range,
            contrast=self.contrast_range,
            saturation=self.saturation_range,
            hue=self.hue_range
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Perform random color jittering on each frame
        for i in range(batch[self.target_key].shape[0]):
            if random.random() < self.probability:
                batch[self.target_key][i] = self.color_jitter_transform(batch[self.target_key][i])

        return batch
