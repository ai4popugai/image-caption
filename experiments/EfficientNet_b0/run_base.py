import os
from typing import Tuple

import torch.nn
from torch import nn
from torch.utils.data import Dataset
from experiments.EfficientNet_b0.efficient_net_b0 import EfficientNet


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

        self.snapshot_dir = os.path.join(os.environ['SNAPSHOT_DIR'], self.project, self.experiment_name, self.run_name)
        self.logs_dir = os.path.join(os.environ['LOGS_DIR'], self.project, self.experiment_name, self.run_name)

        self.optimizer = None
        self.loss = None

        self.metrics = None
        self.train_iters = None
        self.val_iters = None
        self.show_iters = None
        self.max_iteration = None

        self.start_snapshot_path = None

        self.reset_optimizer = None
        self.lr_policy = None
        self.strict_weight_loading = None

        self.cudnn_benchmark = None
        self.device = None

    def setup_model(self) -> nn.Module:
        return EfficientNet()

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        pass

    def train(self,
              reset_optimizer: bool = False,
              start_snapshot: str = None,
              ):
        pass