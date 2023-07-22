from typing import Dict

import torch

from datasets.gpr import NUM_CLASSES
from metricks.base_metric import BaseMetric
from torchmetrics import Accuracy as AccuracyMetric


class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__(name='accuracy')
        self.accuracy = AccuracyMetric(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute accuracy metric.
        :param result: output of the network
        :param batch: batch of data
        :return: accuracy metric
        """
        return self.accuracy(result['logits'], batch['labels'])
