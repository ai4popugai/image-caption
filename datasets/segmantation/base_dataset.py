from typing import Optional

import torch
from torch.utils.data import Dataset


class BaseSegmentationDataset(Dataset):
    """
    Base class for all segmentation datasets

    :param color_map: tensor with BRG colors for corresponded  classes of dataset
    """
    color_map: Optional[torch.Tensor] = None

    def __init__(self):
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item: int):
        raise NotImplementedError
