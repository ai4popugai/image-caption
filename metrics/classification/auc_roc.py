import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing import Dict

from datasets import LABELS_KEY, LOGITS_KEY
from metrics.base_metric import BaseMetric


class AUC_ROC(BaseMetric):
    """METRIC IS BROKEN!!! DON'T USE IT (and drugs)!!!!"""
    def __init__(self, num_classes: int):
        super().__init__(name='AUC-ROC')
        self.num_classes = num_classes
        self.true_labels = []
        self.predicted_probs = []

    def update(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        """
        Accumulate true labels and predicted probabilities.
        :param result: output of the network
        :param batch: batch of data
        """
        self.true_labels.extend(batch[LABELS_KEY].cpu().numpy())
        self.predicted_probs.extend(torch.softmax(result[LOGITS_KEY].detach(), dim=1).cpu().numpy())

    def compute(self) -> float:
        """
        Compute the AUC-ROC metric.
        :return: AUC-ROC score
        """
        true_labels = np.array(self.true_labels)
        true_labels_one_hot = np.zeros((len(true_labels), self.num_classes))
        true_labels_one_hot[np.arange(len(true_labels)), true_labels] = 1

        predicted_probs = np.array(self.predicted_probs)
        roc_auc_per_class = [roc_auc_score(true_labels_one_hot[:, i], predicted_probs[:, i])
                             for i in range(self.num_classes)]
        return sum(roc_auc_per_class) / self.num_classes

    def reset(self):
        """
        Reset the accumulated true labels and predicted probabilities.
        """
        self.true_labels = []
        self.predicted_probs = []
