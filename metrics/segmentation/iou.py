from typing import Dict

import torch
from torchmetrics import JaccardIndex

from datasets import ACTIVATION_MAP_KEY, GROUND_TRUTHS_KEY
from metrics.base_metric import BaseMetric


class IoU(BaseMetric):
    def __init__(self, num_classes):
        super().__init__(name='IoU')
        self.iou_metric = JaccardIndex(task='multiclass', num_classes=num_classes)
        self.unit = '%'
        self.num_samples = 0
        self.total_iou = 0.
        
    def update(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        self.total_iou += self.iou_metric(torch.argmax(result[ACTIVATION_MAP_KEY], dim=1),
                                          batch[GROUND_TRUTHS_KEY])
        self.num_samples += 1

    def compute(self):
        return self.total_iou / self.num_samples * 100

    def reset(self):
        self.total_iou = 0.0
        self.num_samples = 0

    def to(self, device: torch.device) -> None:
        self.iou_metric.to(device)
