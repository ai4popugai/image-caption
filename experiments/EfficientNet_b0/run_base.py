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
from nn_models.classification.base_model import BaseClassificationModel
from normalize.classification.normalize import BatchNormalizer
from train import MODEL_STATE_DICT_KEY, Trainer
from train.run import Run


class RunBase(Run):
    def __init__(self, filename: str):
        super().__init__(filename)

        self._num_classes = NUM_CLASSES
        self.resolution = (256, 256)

        self._normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalizer = BatchNormalizer(normalizer=self._normalizer)

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

    def setup_pretrained_model(self, snapshot_name: str, device: torch.device):
        """
        Script set up model by experiment, run, phase and snapshot.

        :param device: device to map weights
        :param snapshot_name: name of the snapshot from which we take the model
        :return: model with loaded weights.
        """

        model = self.setup_model()

        # load snapshot
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_name)
        checkpoint = torch.load(snapshot_path, map_location=device)
        model.load_state_dict(checkpoint[MODEL_STATE_DICT_KEY], strict=True)

        return model

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        dataset = GPRDataset(resolution=self.resolution)

        # Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
        dataset_size = dataset.__len__()
        validation_size = int(self.validation_split * dataset_size)
        train_size = dataset_size - validation_size

        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
        return train_dataset, validation_dataset
