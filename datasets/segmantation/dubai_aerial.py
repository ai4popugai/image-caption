import os
from typing import Dict

import cv2
import torch
from torchvision.transforms import ToTensor

from datasets import FRAME_KEY, GROUND_TRUTH_KEY
from datasets.segmantation.base_dataset import BaseSegmentationDataset

DUBAI_AERIAL_DATASET = 'DUBAI_AERIAL_DATASET'

COLOR_MAP = {
    0: (152, 16, 60),  # building
    1: (246, 41, 132),  # land
    2: (228, 193, 110),  # road
    3: (58, 221, 254),  # vegetation
    4: (41, 169, 226),  # water
    5: (155, 155, 155)  # unlabeled
}

COLOR_MAP_TENSOR = torch.tensor(list(COLOR_MAP.values()), dtype=torch.uint8)


class DubaiAerial(BaseSegmentationDataset):
    color_map = COLOR_MAP_TENSOR

    def __init__(self):
        super().__init__()
        self.transform = ToTensor()
        if DUBAI_AERIAL_DATASET not in os.environ:
            raise RuntimeError('Dataset root not in environment.')
        self.root = os.environ[DUBAI_AERIAL_DATASET]
        tile_list = [tile for tile in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, tile))]
        self.image_path_list = []
        self.ground_true_path_list = []
        for tile in tile_list:
            tile_path = os.path.join(self.root, tile)
            self.image_path_list += [os.path.join(tile_path, 'images', img)
                                     for img in sorted(os.listdir(os.path.join(tile_path, 'images')))]
            self.ground_true_path_list += [os.path.join(tile_path, 'masks', img)
                                           for img in sorted(os.listdir(os.path.join(tile_path, 'masks')))]
        for (img_path, gt_path) in zip(self.image_path_list, self.ground_true_path_list):
            if os.path.basename(img_path).split('.')[0] != os.path.basename(gt_path).split('.')[0]:
                raise RuntimeError('Dataset is broken! '
                                   'Check out image_path_list and ground_true_path_list correspondence.')

    def __len__(self) -> int:
        return len(self.image_path_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = cv2.imread(self.image_path_list[idx])
        img = self.transform(img)
        gt = cv2.imread(self.ground_true_path_list[idx])
        gt = (self.transform(gt) * 255.).to(torch.uint8)
        return {FRAME_KEY: img, GROUND_TRUTH_KEY: gt}
