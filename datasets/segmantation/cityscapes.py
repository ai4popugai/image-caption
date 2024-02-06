import os
from typing import Dict, Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import Compose, PILToTensor

from datasets import FRAME_KEY, GROUND_TRUTH_KEY, FRAME_T_KEY, FRAME_T_K_KEY, OPTICAL_FLOW_KEY

CITYSCAPES_ROOT = 'CITYSCAPES_DATASET_ROOT'
CITYSCAPES_VIDEO_ROOT = 'CITYSCAPES_VIDEO_DATASET_ROOT'
HEIGHT = 1024
WIDTH = 2048
CHANNELS = 3

MAP_34_TO_19 = torch.tensor([19, 19, 19, 19, 19, 19, 19, 0, 1, 19, 19, 2, 3, 4, 19, 19, 19, 5,
                             19, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 19, 16, 17, 18, 19])

COLOR_MAP_20 = {
    0: (128, 64, 128),
    1: (232, 35, 244),
    2: (70, 70, 70),
    3: (156, 102, 102),
    4: (153, 153, 190),
    5: (153, 153, 153),
    6: (30, 170, 250),
    7: (0, 220, 220),
    8: (35, 142, 107),
    9: (152, 251, 152),
    10: (180, 130, 70),
    11: (60, 20, 220),
    12: (0, 0, 255),
    13: (142, 0, 0),
    14: (70, 0, 0),
    15: (100, 60, 0),
    16: (100, 80, 0),
    17: (230, 0, 0),
    18: (32, 11, 119),
    19: (0, 0, 0)}  # void backgrounds

COLOR_MAP_19_TENSOR = torch.tensor(list(COLOR_MAP_20.values()), dtype=torch.uint8)

COLOR_MAP_34 = {
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
COLOR_MAP_34_TENSOR = torch.tensor(list(COLOR_MAP_34.values()), dtype=torch.uint8)

PIXELS_PER_CLASSES_34 = torch.tensor([6.7643e+06, 1.5170e+05, 1.6685e+05, 2.1188e+05, 2.9095e+05, 4.3800e+05,
                                      9.0043e+05, 1.5947e+06, 2.9845e+06, 4.9343e+06, 7.7613e+06, 1.2875e+07,
                                      1.7736e+07, 2.6805e+07, 3.6319e+07, 4.3691e+07, 6.1121e+07, 6.9351e+07,
                                      8.0350e+07, 9.5971e+07, 1.0636e+08, 1.1570e+08, 1.2419e+08, 1.3330e+08,
                                      1.4437e+08, 1.4055e+08, 1.5200e+08, 1.5549e+08, 1.5921e+08, 1.7154e+08,
                                      1.6522e+08, 1.7315e+08, 1.7322e+08, 1.7377e+08], dtype=torch.float64)

CLASSES_WEIGHTS_34 = torch.tensor([5.6972e-03, 2.5404e-01, 2.3097e-01, 1.8188e-01, 1.3245e-01, 8.7985e-02,
                                   4.2799e-02, 2.4166e-02, 1.2913e-02, 7.8101e-03, 4.9653e-03, 2.9932e-03,
                                   2.1728e-03, 1.4377e-03, 1.0611e-03, 8.8205e-04, 6.3051e-04, 5.5569e-04,
                                   4.7962e-04, 4.0155e-04, 3.6233e-04, 3.3308e-04, 3.1031e-04, 2.8910e-04,
                                   2.6694e-04, 2.7419e-04, 2.5354e-04, 2.4785e-04, 2.4205e-04, 2.2466e-04,
                                   2.3325e-04, 2.2257e-04, 2.2248e-04, 2.2177e-04], dtype=torch.float32)


def setup_opt_flow_path(step: int, optical_flow_model: str, optical_flow_model_kwargs: Dict[str, Any]):
    dst_path = os.path.join(os.environ[CITYSCAPES_VIDEO_ROOT], 'optical_flow')
    dst_path = f'{dst_path}_step-{step}_model-{optical_flow_model}'
    for key, val in optical_flow_model_kwargs.items():
        dst_path += f'_{key}-{val}'
    return dst_path


class CityscapesVideoDataset(Dataset):
    color_map = COLOR_MAP_34_TENSOR

    def __init__(self, step: int = 1):
        """
        Cityscapes video dataset wrapper.

        :param step: distance between 2 frames in __getitem__ method. Default: 1 frame.
        3-d part consists of the first 50 images from 0-d part.
        """
        super().__init__()
        if CITYSCAPES_VIDEO_ROOT not in os.environ:
            raise RuntimeError(
                f'Failed to init cityscapes dataset instance: {CITYSCAPES_VIDEO_ROOT} not in environment.')
        self.root = os.path.join(os.environ[CITYSCAPES_VIDEO_ROOT], 'leftImg8bit', 'demoVideo')
        self.step = step
        self.frame_transforms = Compose([
            PILToTensor(),
        ])

        cities_path_list = [os.path.join(self.root, city) for city in sorted(os.listdir(self.root))
                            if os.path.isdir(os.path.join(self.root, city))]
        self.frames_list = []
        self.indexes = []  # list to map inner index to index in frames_list
        for city_path in cities_path_list:
            frames_names = sorted(os.listdir(city_path))
            self.frames_list += [os.path.join(city_path, frame_name) for frame_name in frames_names]
            self.indexes += list(range(len(self.frames_list) - len(frames_names), len(self.frames_list) - self.step))

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        idx = self.indexes[idx]
        frame_t = Image.open(self.frames_list[idx]).convert("RGB")
        frame_t = self.frame_transforms(frame_t).byte()

        frame_t_k = Image.open(self.frames_list[idx + self.step]).convert("RGB")
        frame_t_k = self.frame_transforms(frame_t_k).byte()
        return {FRAME_T_KEY: frame_t,
                FRAME_T_K_KEY: frame_t_k,
                }


class CityscapesVideoOptFlowDataset(Dataset):
    def __init__(self, optical_flow_model: str, optical_flow_model_kwargs: Dict[str, Any], step: int = 1):
        super().__init__()
        opt_flow_path = setup_opt_flow_path(step, optical_flow_model, optical_flow_model_kwargs)
        self.dataset = CityscapesVideoDataset(step=step)
        self.optical_flow_path_list = [os.path.join(opt_flow_path, ff) for ff in sorted(os.listdir(opt_flow_path))
                                       if ff.endswith('.pt')]
        if len(self.dataset) != len(self.optical_flow_path_list):
            raise RuntimeError('Incompatible amount of precomputed optical flow maps for given '
                               'optical_flow_model and optical_flow_model_kwargs')

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        item = self.dataset[idx]
        item[OPTICAL_FLOW_KEY] = torch.load(self.optical_flow_path_list[idx])
        return item


class CityscapesDataset34(Dataset):
    color_map = COLOR_MAP_34_TENSOR

    def __init__(self, split: str, mode: str = 'fine'):
        if CITYSCAPES_ROOT not in os.environ:
            raise RuntimeError(f'Failed to init cityscapes dataset instance: {CITYSCAPES_ROOT} not in environment.')
        self.transform = Compose([
            PILToTensor(),
        ])
        self.dataset = Cityscapes(os.path.join(os.environ[CITYSCAPES_ROOT], mode), split, mode, target_type='semantic',
                                  transform=self.transform, target_transform=self.transform)
        super().__init__()

    def __len__(self) -> int:
        return self.dataset.__len__()

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        frame, segmentation = self.dataset.__getitem__(idx)
        frame = frame.to(torch.float32) / 255.
        return {FRAME_KEY: torch.flip(frame, [0]), GROUND_TRUTH_KEY: segmentation.squeeze(0).to(torch.int64)}
        # convert frames from RGB to BGR


class CityscapesDataset19(Dataset):
    color_map = COLOR_MAP_19_TENSOR

    def __init__(self, split: str, mode: str = 'fine'):
        self.dataset = CityscapesDataset34(split, mode)
        super().__init__()

    def __len__(self) -> int:
        return self.dataset.__len__()

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        item = self.dataset.__getitem__(idx)
        item[GROUND_TRUTH_KEY] = MAP_34_TO_19[item[GROUND_TRUTH_KEY]]
        return item
