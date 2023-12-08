import math
import random
from abc import abstractmethod, ABC
from typing import Dict, Tuple, Union, List

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import hflip
import torchvision.transforms.functional as F

from datasets import GROUND_TRUTHS_KEY, LABELS_KEY


class BaseAug(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, frames: torch.Tensor):
        pass


class ColorAug(BaseAug):
    """AUG IS BROKEN!!! DON'T USE IT (and drugs)!!!!"""
    def __init__(self, b_low: float = 0.7, s_low: float = 0.7, c_low: float = 0.7):
        super().__init__()
        self.transform = transforms.ColorJitter(brightness=(b_low, 1.),
                                                saturation=(s_low, 1.), contrast=(c_low, 1.))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch[GROUND_TRUTHS_KEY] = self.transform(batch[GROUND_TRUTHS_KEY])
        return batch


class RandomFlip(BaseAug):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            batch[GROUND_TRUTHS_KEY] = hflip(batch[GROUND_TRUTHS_KEY])
        return batch


class RandomCrop(BaseAug):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch[GROUND_TRUTHS_KEY] = transforms.RandomCrop(self.size)(batch[GROUND_TRUTHS_KEY])
        return batch


class RandomResizedCropWithProb(BaseAug):
    def __init__(self, size: Union[List[float], Tuple[int, int]],
                 probability: float = 0.5):
        super().__init__()
        self.size = size
        self.probability = probability

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if isinstance(self.size[0], float):
            change_factor = math.sqrt(random.uniform(self.size[0], self.size[1]))
            new_height = int(batch[GROUND_TRUTHS_KEY].shape[-2] * change_factor)
            new_width = int(batch[GROUND_TRUTHS_KEY].shape[-1] * change_factor)
            size = (new_height, new_width)
        else:
            size = self.size

        transform = transforms.Compose([
            transforms.RandomCrop(size),
            transforms.Resize(batch[GROUND_TRUTHS_KEY].shape[-2:], antialias=False)  # Resize back to original resolution
        ])

        # Perform random resized crop on each frame
        for i in range(batch[GROUND_TRUTHS_KEY].shape[0]):
            if random.random() < self.probability:
                batch[GROUND_TRUTHS_KEY][i] = transform(batch[GROUND_TRUTHS_KEY][i])

        return batch


class CenterCrop(BaseAug):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch[GROUND_TRUTHS_KEY] = transforms.CenterCrop(self.size)(batch[GROUND_TRUTHS_KEY])
        return batch


class Rotate(BaseAug):
    def __init__(self, angle_range=(-180, 180)):
        super().__init__()
        self.angle_range = angle_range

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Perform rotation on each frame
        for i in range(batch[GROUND_TRUTHS_KEY].shape[0]):
            # Randomly select an angle within the defined range
            angle = torch.FloatTensor(1).uniform_(self.angle_range[0], self.angle_range[1]).item()
            batch[GROUND_TRUTHS_KEY][i] = self.rotate_frame(batch[GROUND_TRUTHS_KEY][i], angle)

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
    def __init__(self, angle_range=(-180, 180), probability: float = 0.5):
        super().__init__()
        self.angle_range = angle_range
        self.probability = probability

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Perform rotation on each frame
        for i in range(batch[GROUND_TRUTHS_KEY].shape[0]):
            if random.random() < self.probability:
                # Randomly select an angle within the defined range
                angle = torch.FloatTensor(1).uniform_(self.angle_range[0], self.angle_range[1]).item()
                batch[GROUND_TRUTHS_KEY][i] = Rotate.rotate_frame(batch[GROUND_TRUTHS_KEY][i], angle)

        return batch


class RandomColorJitterWithProb(BaseAug):
    def __init__(
        self,
        probability: float = 0.5,
        brightness_range: Tuple[float, float] = (1, 1),
        contrast_range: Tuple[float, float] = (1, 1),
        saturation_range: Tuple[float, float] = (1, 1),
        hue_range: Tuple[float, float] = (0, 0)
    ):
        super().__init__()
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
        for i in range(batch[GROUND_TRUTHS_KEY].shape[0]):
            if random.random() < self.probability:
                batch[GROUND_TRUTHS_KEY][i] = self.color_jitter_transform(batch[GROUND_TRUTHS_KEY][i])

        return batch
