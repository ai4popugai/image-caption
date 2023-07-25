import os
from typing import Tuple

import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor

NUM_CLASSES = 1200


class GPRDataset(Dataset):
    def __init__(self, resolution: Tuple[int, int] = (64, 64)):
        if os.getenv("GPR_DATASET") is None:
            raise RuntimeError('Dataset path must be set up.')
        root = os.environ['GPR_DATASET']
        self.resolution = resolution
        self.frames_list = [os.path.join(root, file_name) for file_name in sorted(os.listdir(root))]
        self.frame_transforms = Compose([
            ToTensor(),
            Resize(resolution, InterpolationMode.BILINEAR)
        ])

    def __len__(self) -> int:
        return len(self.frames_list)

    def __getitem__(self, idx):
        frame = cv2.imread(self.frames_list[idx], cv2.IMREAD_COLOR)
        frame = self.frame_transforms(frame)
        return {'frames': frame,
                'labels': int(os.path.basename(self.frames_list[idx]).split('_')[0])}
