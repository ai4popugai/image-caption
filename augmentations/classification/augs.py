from abc import abstractmethod, ABC
from typing import Dict, Tuple

import cv2
import numpy as np
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
    def __init__(self, b_low: float = 0.7, s_low: float = 0.7, c_low: float = 0.7):
        super().__init__()
        self.transform = transforms.ColorJitter(brightness=(b_low, 1.),
                                                saturation=(s_low, 1.), contrast=(c_low, 1.))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformed_frames = self.transform(batch['frames'])
        return {'frames': transformed_frames, 'labels': batch['labels']}


class RandomFlip(BaseAug):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformed_frames = batch['frames']
        if torch.rand(1).item() < self.p:
            transformed_frames = hflip(transformed_frames)
        return {'frames': transformed_frames, 'labels': batch['labels']}


class RandomCrop(BaseAug):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformed_frames = transforms.RandomCrop(self.size)(batch['frames'])
        return {'frames': transformed_frames, 'labels': batch['labels']}


class CenterCrop(BaseAug):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformed_frames = transforms.CenterCrop(self.size)(batch['frames'])
        return {'frames': transformed_frames, 'labels': batch['labels']}


class Rotate(BaseAug):
    def __init__(self, angle_range=(-180, 180)):
        super().__init__()
        self.angle_range = angle_range

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformed_frames = batch['frames']

        # Randomly select an angle within the defined range
        angle = torch.FloatTensor(1).uniform_(self.angle_range[0], self.angle_range[1]).item()

        # Perform rotation on each frame
        for i in range(transformed_frames.shape[0]):
            transformed_frames[i] = self.rotate_frame(transformed_frames[i], angle)

        return {'frames': transformed_frames, 'labels': batch['labels']}

    @staticmethod
    def rotate_frame(frame, angle):
        # Convert frame to numpy array
        frame = frame.numpy()

        height, width = frame.shape[:2]
        center = (width / 2, height / 2)

        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the actual rotation
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

        # Convert back to torch.Tensor
        rotated_frame = torch.from_numpy(rotated_frame)

        return rotated_frame
