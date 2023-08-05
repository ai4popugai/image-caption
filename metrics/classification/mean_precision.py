from typing import Dict

import torch
from torchmetrics import Precision as PrecisionMetric

from datasets import LABELS_KEY, LOGITS_KEY
from metrics.base_metric import BaseMetric


class MeanPrecision(BaseMetric):
    def __init__(self, num_classes: int):
        super().__init__(name='mean_precision')
        self.precision = PrecisionMetric(average='macro', task='multiclass', num_classes=num_classes)
        self.num = 0
        self.prob_precision = 0.0
        self.unit = '%'

    def update(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        """
        Compute mean precision metric.
        :param result: output of the network
        :param batch: batch of data
        :return: mean precision metric
        """
        self.num += 1
        self.prob_precision += self.precision(result[LOGITS_KEY].cpu(), batch[LABELS_KEY].cpu()).item()

    def compute(self) -> float:
        return (self.prob_precision / self.num) * 100

    def reset(self):
        self.prob_precision = 0.0
        self.num = 0

    def to(self, device: torch.device) -> None:
        self.precision.to(device)
