import os
from typing import Tuple, Dict

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from datasets import FRAME_KEY, LABEL_KEY


class DogsCatsDataset(Dataset):
    def __init__(self, mode: str, resolution: Tuple[int, int] = (400, 400)):
        if os.getenv("DOGS_CATS_DATASET") is None:
            raise RuntimeError('Dataset path must be set up.')
        if mode != 'train' and mode != 'test':
            raise RuntimeError('Dataset mode must be "train" or "test"')
        self.resolution = resolution
        self.mode = mode if mode == 'train' else mode + '1'
        self.root = os.path.join(os.environ['DOGS_CATS_DATASET'], mode)
        self.img_files = [os.path.join(self.root, file) for file in sorted(os.listdir(self.root))]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.resolution, T.InterpolationMode.BILINEAR, antialias=False)
        ])

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file = self.img_files[idx]
        label = 0 if os.path.basename(file).split('.')[0] == 'cat' else 1
        return {FRAME_KEY: self.transform(cv2.imread(file)), LABEL_KEY: torch.tensor(label, dtype=torch.int64)}
