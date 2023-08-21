import os
from typing import Tuple

from torch.utils.data import Dataset

from augmentations.classification.augs import RandomFlip, RandomCrop, Rotate, RandomResizedCropWithProb
from datasets.classification.gpr import GPRDataset
from experiments.EfficientNet_b0.run_base import RunBase


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'lr': 6e-4, 'weight_decay': 3e-3}
        self.lr_policy = None

        self.train_augs = [RandomResizedCropWithProb(size=self.crop_size, probability=0.5),
                           RandomFlip(), Rotate()]
        self.val_augs = None

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Provide new train - test 80/20 splits.
        :return: train and val datasets
        """
        dataset = GPRDataset(resolution=self.resolution)

        train_dataset, val_dataset = GPRDataset(resolution=self.resolution), GPRDataset(resolution=self.resolution)
        train_dataset.frames_list = []
        val_dataset.frames_list = []

        for i in range(0, len(dataset), 10):
            files_slice = dataset.frames_list[i: i + 10]
            train_dataset.frames_list += files_slice[:8]
            val_dataset.frames_list += files_slice[8:]
        del dataset
        return train_dataset, val_dataset


if __name__ == '__main__':
    Phase().train(start_snapshot=None)
