import os
from typing import Tuple, List, Dict

import torch.optim.optimizer
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from augmentations.classification.augs import ColorAug, RandomFlip, RandomCrop, CenterCrop
from datasets.classification.gpr import GPRDataset, NUM_CLASSES
from experiments.EfficientNet_b0.efficient_net_b0 import EfficientNet
from loss.classification.cross_entropy import CrossEntropyLoss
from metrics.classification.accuracy import Accuracy
from metrics.classification.mean_precision import MeanPrecision
from metrics.classification.mean_recall import MeanRecall
from models.classification.base_model import BaseClassificationModel
from normalize.classification.normalize import BatchNormalizer
from train.run import Run


class RunBase(Run):
    def __init__(self, filename: str):
        super().__init__(filename)

        self._num_classes = NUM_CLASSES
        self.resolution = (256, 256)
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize_batch = BatchNormalizer(normalizer=self.normalizer)

        self.batch_size = 64
        self.num_workers = 8

        self.validation_split = 0.2

        self.optimizer_class = torch.optim.Adam
        self.loss = CrossEntropyLoss()

        self.train_metrics = [Accuracy(self._num_classes),
                              MeanRecall(self._num_classes), MeanPrecision(self._num_classes)]
        self.val_metrics = [Accuracy(self._num_classes),
                            MeanRecall(self._num_classes), MeanPrecision(self._num_classes)]

        self.crop_size = (192, 192)
        self.train_augs = [RandomFlip(), RandomCrop(self.crop_size)]
        self.val_augs = [CenterCrop(self.crop_size)]

        self.start_snapshot = None

        self.lr_policy = None

    def setup_model(self) -> BaseClassificationModel:
        return EfficientNet(num_classes=self._num_classes)

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        dataset = GPRDataset(resolution=self.resolution)

        # Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
        dataset_size = dataset.__len__()
        validation_size = int(self.validation_split * dataset_size)
        train_size = dataset_size - validation_size

        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
        return train_dataset, validation_dataset
