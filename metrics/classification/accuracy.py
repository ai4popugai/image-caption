from typing import Dict

import torch

from datasets import LABEL_KEY, LOGIT_KEY
from metrics.base_metric import BaseMetric
from torchmetrics import Accuracy as AccuracyMetric


class Accuracy(BaseMetric):
    def __init__(self, num_classes: int):
        super().__init__(name='accuracy')
        self.accuracy = AccuracyMetric(task="multiclass", num_classes=num_classes)
        self.num = 0
        self.prob_acc = 0.0
        self.unit = '%'

    def update(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        """
        Compute accuracy metric.
        :param result: output of the network
        :param batch: batch of data
        :return: accuracy metric
        """
        self.num += 1
        self.prob_acc += self.accuracy(result[LOGIT_KEY].cpu(), batch[LABEL_KEY].cpu()).item()

    def compute(self) -> float:
        return (self.prob_acc / self.num) * 100

    def reset(self):
        self.prob_acc = 0.0
        self.num = 0

    def to(self, device: torch.device) -> None:
        self.accuracy.to(device)
