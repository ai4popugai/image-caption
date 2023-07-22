import os
from typing import Tuple, List

from torch import nn
from torch.utils.data import Dataset, random_split

from datasets.classification.gpr import GPRDataset
from experiments.EfficientNet_b0.efficient_net_b0 import EfficientNet
from metricks.classification.accuracy import Accuracy
from metricks.base_metric import BaseMetric


class RunBase:
    def __init__(self, filename: str):
        self.name = os.path.splitext(os.path.basename(filename))[0]  # i.e. phase_1
        run_path = os.path.split(filename)[0]
        self.run_name = os.path.basename(run_path)  # i.e. run_10
        experiment_path = os.path.split(run_path)[0]  # i.e. proj/experiments/exp_name
        self.experiment_name = os.path.basename(experiment_path)  # i.e. wav2lip3
        self.project = os.path.basename(os.path.split(os.path.split(experiment_path)[0])[0])  # i.e. wav2lip

        self.batch_size = 64
        self.num_workers = 8

        self.validation_split = 0.2

        self.snapshot_dir = os.path.join(os.environ['SNAPSHOT_DIR'], self.project, self.experiment_name, self.run_name)
        self.logs_dir = os.path.join(os.environ['LOGS_DIR'], self.project, self.experiment_name, self.run_name)

        self.optimizer = None
        self.loss = nn.CrossEntropyLoss()

        self.metrics: List[BaseMetric] = [Accuracy()]
        self.train_iters = 100
        self.show_iters = 10
        self.snapshot_iters = 200
        self.max_iteration = 1000000

        self.start_snapshot_path = None

        self.reset_optimizer = True
        self.lr_policy = None
        self.strict_weight_loading = True

        self.cudnn_benchmark = None
        self.device = None

    def setup_model(self) -> nn.Module:
        return EfficientNet()

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        dataset = GPRDataset(resolution=(64, 64))

        # Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
        dataset_size = dataset.__len__()
        validation_size = int(self.validation_split * dataset_size)
        train_size = dataset_size - validation_size

        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
        return train_dataset, validation_dataset

    def train(self,
              reset_optimizer: bool = False,
              start_snapshot: str = None,
              ):
        pass