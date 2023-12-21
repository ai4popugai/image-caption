import torch
from torch.utils.data import Dataset


class BaseSegmentationDataset(Dataset):
    def __init__(self, color_map: torch.Tensor):
        """
        Base class for all segmentation datasets

        :param color_map: tensor with BRG colors for corresponded  classes of dataset
        """
        super().__init__()
        self.color_map = color_map

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item: int):
        raise NotImplementedError
