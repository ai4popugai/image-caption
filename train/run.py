import os
from abc import ABC
from typing import Tuple

from torch.utils.data import Dataset


class Run(ABC):
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
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # optimizer
        self.optimizer = None
        self.reset_optimizer = None
        self.lr_policy = None

        # loss
        self.loss = None

        # list of metrics
        self.metrics: None

        # num of iterations
        self.train_iters = 100
        self.show_iters = 10
        self.snapshot_iters = 200
        self.max_iteration = 1000000

        # snapshots
        self.start_snapshot_name = None
        self.strict_weight_loading = True

        # cudnn
        self.cudnn_benchmark = True

    def setup_model(self):
        raise NotImplementedError

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def train(self,
              reset_optimizer: bool = False,
              start_snapshot: str = None,
              ):
        pass