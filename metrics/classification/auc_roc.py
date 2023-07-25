import torch
from sklearn.metrics import roc_auc_score
from typing import Dict

from metrics.base_metric import BaseMetric


class AUC_ROC(BaseMetric):
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
        self.true_labels.extend(batch['labels'].cpu().numpy())
        self.predicted_probs.extend(torch.softmax(result['logits'].detach(), dim=1).cpu().numpy())

    def compute(self) -> float:
        """
        Compute the AUC-ROC metric.
        :return: AUC-ROC score
        """
        true_labels = torch.tensor(self.true_labels)
        predicted_probs = torch.tensor(self.predicted_probs)
        auc_roc_per_class = [roc_auc_score(true_labels[:, i], predicted_probs[:, i]) for i in range(self.num_classes)]
        return sum(auc_roc_per_class) / self.num_classes

    def reset(self):
        """
        Reset the accumulated true labels and predicted probabilities.
        """
        self.true_labels = []
        self.predicted_probs = []
