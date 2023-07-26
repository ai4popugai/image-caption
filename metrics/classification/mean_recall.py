from typing import Dict

import torch
from torchmetrics import Recall as RecallMetric

from metrics.base_metric import BaseMetric


class MeanRecall(BaseMetric):
    def __init__(self, num_classes: int):
        super().__init__(name='mean_recall')
        self.recall = RecallMetric(average='macro', task='multiclass', num_classes=num_classes)
        self.num = 0
        self.prob_recall = 0.0

    def update(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        """
        Compute mean recall metric.
        :param result: output of the network
        :param batch: batch of data
        :return: mean recall metric
        """
        self.num += 1
        self.prob_recall += self.recall(result['logits'], batch['labels']).item()

    def compute(self) -> float:
        return self.prob_recall / self.num

    def reset(self):
        self.prob_recall = 0.0
        self.num = 0
