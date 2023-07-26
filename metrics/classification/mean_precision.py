from typing import Dict

import torch
from torchmetrics import Precision as PrecisionMetric

from metrics.base_metric import BaseMetric


class MeanPrecision(BaseMetric):
    def __init__(self, num_classes: int):
        super().__init__(name='mean_precision')
        self.precision = PrecisionMetric(average='macro', task='multiclass', num_classes=num_classes)
        self.num = 0
        self.prob_precision = 0.0

    def update(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        """
        Compute mean precision metric.
        :param result: output of the network
        :param batch: batch of data
        :return: mean precision metric
        """
        self.num += 1
        self.prob_precision += self.precision(result['logits'], batch['labels']).item()

    def compute(self) -> float:
        return self.prob_precision / self.num

    def reset(self):
        self.prob_precision = 0.0
        self.num = 0
