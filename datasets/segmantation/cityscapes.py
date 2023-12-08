import os
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import Compose, PILToTensor, Resize, InterpolationMode

CITYSCAPES_ROOT = 'CITYSCAPES_DATASET_ROOT'
CITYSCAPES_VIDEO_ROOT = 'CITYSCAPES_VIDEO_DATASET_ROOT'
HEIGHT = 1024
WIDTH = 2048
CHANNELS = 3
NUM_CLASSES = 34

COLOR_MAP = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (0, 74, 111),
    6: (81, 0, 81),
    7: (128, 64, 128),
    8: (232, 35, 244),
    9: (160, 170, 250),
    10: (140, 150, 230),
    11: (70, 70, 70),
    12: (156, 102, 102),
    13: (153, 153, 190),
    14: (180, 165, 180),
    15: (100, 100, 150),
    16: (90, 120, 150),
    17: (153, 153, 153),
    18: (153, 153, 153),
    19: (30, 170, 250),
    20: (0, 220, 220),
    21: (35, 142, 107),
    22: (152, 251, 152),
    23: (180, 130, 70),
    24: (60, 20, 220),
    25: (0, 0, 255),
    26: (142, 0, 0),
    27: (70, 0, 0),
    28: (100, 60, 0),
    29: (90, 0, 0),
    30: (110, 0, 0),
    31: (100, 80, 0),
    32: (230, 0, 0),
    33: (32, 11, 119),
}

PIXELS_PER_CLASSES = torch.tensor([6.7643e+06, 1.5170e+05, 1.6685e+05, 2.1188e+05, 2.9095e+05, 4.3800e+05,
                                   9.0043e+05, 1.5947e+06, 2.9845e+06, 4.9343e+06, 7.7613e+06, 1.2875e+07,
                                   1.7736e+07, 2.6805e+07, 3.6319e+07, 4.3691e+07, 6.1121e+07, 6.9351e+07,
                                   8.0350e+07, 9.5971e+07, 1.0636e+08, 1.1570e+08, 1.2419e+08, 1.3330e+08,
                                   1.4437e+08, 1.4055e+08, 1.5200e+08, 1.5549e+08, 1.5921e+08, 1.7154e+08,
                                   1.6522e+08, 1.7315e+08, 1.7322e+08, 1.7377e+08], dtype=torch.float64)

CLASSES_WEIGHTS = torch.tensor([5.6972e-03, 2.5404e-01, 2.3097e-01, 1.8188e-01, 1.3245e-01, 8.7985e-02,
                                4.2799e-02, 2.4166e-02, 1.2913e-02, 7.8101e-03, 4.9653e-03, 2.9932e-03,
                                2.1728e-03, 1.4377e-03, 1.0611e-03, 8.8205e-04, 6.3051e-04, 5.5569e-04,
                                4.7962e-04, 4.0155e-04, 3.6233e-04, 3.3308e-04, 3.1031e-04, 2.8910e-04,
                                2.6694e-04, 2.7419e-04, 2.5354e-04, 2.4785e-04, 2.4205e-04, 2.2466e-04,
                                2.3325e-04, 2.2257e-04, 2.2248e-04, 2.2177e-04], dtype=torch.float32)

COLOR_MAP_TENSOR = torch.tensor(list(COLOR_MAP.values()), dtype=torch.uint8)


def map_to_classes(seg: torch.Tensor) -> torch.Tensor:
    """
    Method converts activation map [batch_size, N_CLASSES, h, w] to tensor with class label in dim 1
    [batch_size, CHANNELS, h, w].
    :param seg: input tensor
    :return: tensor with class labels
    """
    # softmax = torch.nn.Softmax(dim=1)
    segmentations = torch.argmax(seg, dim=1).unsqueeze(dim=1)
    return segmentations.repeat(1, CHANNELS, 1, 1)


def classes_to_colors(seg: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor [batch_size, CHANNELS, h, w] with classes in 1 dim
    to tensor [batch_size, CHANNELS, h, w] with colors in 1 dim.
    :param seg: input tensor.
    :return: color tensor.
    """
    mapped = COLOR_MAP_TENSOR[seg.select(1, 0)].permute(0, 3, 1, 2)
    return mapped


def map_to_colors(seg: torch.Tensor) -> torch.Tensor:
    """
    Method converts activation map [batch_size, N_CLASSES, h, w]
    to tensor [batch_size, CHANNELS, h, w] with colors in 1 dim.
    :param seg:
    :return:
    """
    return classes_to_colors(map_to_classes(seg))


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.uint8)).unsqueeze(dim=0).repeat(CHANNELS, 1, 1)


class CityScapesVideoDataset(Dataset):
    def __init__(self, part: int, resolution: Tuple[int, int] = (1024, 2048)):
        """
        Cityscapes video dataset wrapper.

        :param part: must be 0, 1 or 2.
        3-d part consists of the first 50 images from 0-d part.
        """
        if part != 0 and part != 1 and part != 2 and part != 3:
            raise RuntimeError('Dataset part must be 0, 1 or 2')
        if CITYSCAPES_VIDEO_ROOT not in os.environ:
            raise RuntimeError(
                f'Failed to init cityscapes dataset instance: {CITYSCAPES_VIDEO_ROOT} not in environment.')
        self.root = os.path.join(os.environ[CITYSCAPES_VIDEO_ROOT], 'leftImg8bit', 'demoVideo')
        self.city_name = 'stuttgart'
        self.frames_list = []
        self.frame_transforms = Compose([
            PILToTensor(),
            Resize(resolution, InterpolationMode.NEAREST)
        ])

        city_dir = os.path.join(self.root, f'{self.city_name}_{"%02d" % part}')
        for frame_name in sorted(os.listdir(city_dir)):
            self.frames_list.append(os.path.join(city_dir, frame_name))

    def __len__(self) -> int:
        return len(self.frames_list)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        frame = Image.open(self.frames_list[idx]).convert("RGB")
        frame = self.frame_transforms(frame).byte()
        return {'frame': frame}


class CityScapesWrapper(Cityscapes):
    def __init__(self, mode: str, split: str, resolution: Tuple[int, int] = (1024, 2048)):
        if CITYSCAPES_ROOT not in os.environ:
            raise RuntimeError(f'Failed to init cityscapes dataset instance: {CITYSCAPES_ROOT} not in environment.')
        target_transform = MaskToTensor()
        self.frame_transforms = Compose([
            PILToTensor(),
            Resize(resolution, InterpolationMode.NEAREST)
        ])
        self.seg_transforms = Compose([
            Resize(resolution, InterpolationMode.NEAREST)
        ])
        super().__init__(os.path.join(os.environ[CITYSCAPES_ROOT], mode), split, mode, target_type='semantic',
                         transform=self.frame_transforms, target_transform=target_transform)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        frame, segmentation = super().__getitem__(idx)
        frame = frame.byte()
        segmentation = self.seg_transforms(segmentation)
        return {'frame': frame, 'seg': segmentation}
