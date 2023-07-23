from typing import Dict

import torch

from metricks.base_metric import BaseMetric
from torchmetrics import Accuracy as AccuracyMetric


class Accuracy(BaseMetric):
    def __init__(self, num_classes: int):
        super().__init__(name='accuracy')
        self.accuracy = AccuracyMetric(task="multiclass", num_classes=num_classes)
        self.num = 0
        self.prob_acc = 0.0

    def update(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        """
        Compute accuracy metric.
        :param result: output of the network
        :param batch: batch of data
        :return: accuracy metric
        """
        self.num += 1
        self.prob_acc += self.accuracy(result['logits'], batch['labels']).item()

    def compute(self) -> float:
        return self.prob_acc / self.num

    def reset(self):
        self.prob_acc = 0.0
        self.num = 0
