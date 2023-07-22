import torch

from datasets.gpr import NUM_CLASSES
from metricks.base_metric import BaseMetric
from torchmetrics import Accuracy as AccuracyMetric


class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__(name='accuracy')
        self.accuracy = AccuracyMetric(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute accuracy metric.
        :param preds: logits from model
        :param targets: real labels
        :return: accuracy metric
        """
        return self.accuracy(preds, targets)
