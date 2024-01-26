import os
from typing import Tuple, Dict

import torch.optim.optimizer
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from augmentations.augs import RandomFlip
from datasets import LOGIT_KEY, LABEL_KEY, FRAME_KEY
from datasets.classification.dogs_cats import DogsCatsDataset
from nn_models.classification.efficient_net.efficient_net_b0 import EfficientNet
from loss.cross_entropy import CrossEntropyLoss
from metrics.classification.accuracy import Accuracy
from metrics.classification.mean_precision import MeanPrecision
from metrics.classification.mean_recall import MeanRecall
from nn_models.classification.base_model import BaseClassificationModel
from normalize.normalize import BatchNormalizer
from train.run import Run
from transforms.segmentration import BaseToImageTransforms, FramesToImage


class RunBase(Run):
    def __init__(self, filename: str):
        super().__init__(filename)

        self._num_classes = 2

        self._normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalizer = BatchNormalizer(normalizer=self._normalizer, target_key=FRAME_KEY)

        self.batch_size = 16
        self.num_workers = 0

        self.train_iters = 50
        self.snapshot_iters = 50
        self.show_iters = 10
        self.accum_iters = 2

        self.validation_split = 0.2

        self.optimizer_class = torch.optim.Adam
        self.loss = CrossEntropyLoss(result_trg_key=LOGIT_KEY, batch_trg_key=LABEL_KEY)

        self.train_metrics = [Accuracy(self._num_classes),
                              MeanRecall(self._num_classes), MeanPrecision(self._num_classes)]
        self.val_metrics = [Accuracy(self._num_classes),
                            MeanRecall(self._num_classes), MeanPrecision(self._num_classes)]

        self.train_augs = [RandomFlip()]

        self.start_snapshot = None

        self.lr_policy = None

        self.batch_dump_flag = False

    def setup_model(self) -> BaseClassificationModel:
        return EfficientNet(num_classes=self._num_classes, pretrain=True)

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        dataset = DogsCatsDataset(mode='train')

        # Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
        dataset_size = dataset.__len__()
        validation_size = int(self.validation_split * dataset_size)
        train_size = dataset_size - validation_size

        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
        return train_dataset, validation_dataset

    def get_batch_sample_to_image_map(self) -> Dict[str, BaseToImageTransforms]:
        return {FRAME_KEY: FramesToImage()}

