import os
from abc import ABC
from typing import Tuple, Optional, List

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LRScheduler

from augmentations.classification.augs import BaseAug
from metricks.base_metric import BaseMetric
from train.train import Trainer


class Run(ABC):
    def __init__(self, filename: str):
        self.name = os.path.splitext(os.path.basename(filename))[0]  # i.e. phase_1
        run_path = os.path.split(filename)[0]
        self.run_name = os.path.basename(run_path)  # i.e. run_10
        experiment_path = os.path.split(run_path)[0]  # i.e. proj/experiments/exp_name
        self.experiment_name = os.path.basename(experiment_path)  # i.e. wav2lip3
        self.project = os.path.basename(os.path.split(os.path.split(experiment_path)[0])[0])  # i.e. wav2lip

        self.batch_size: int = 64
        self.num_workers: int = 8

        self.validation_split: float = 0.2

        # num of iterations
        self.train_iters: int = 100
        self.show_iters: int = 10
        self.snapshot_iters: int = 200
        self.max_iteration: int = 1000000

        self.snapshot_dir: str = os.path.join(os.environ['SNAPSHOT_DIR'], self.project, self.experiment_name, self.run_name)
        self.logs_dir: str = os.path.join(os.environ['LOGS_DIR'], self.project, self.experiment_name, self.run_name)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # optimizer
        self.optimizer: Optional[Optimizer] = None
        self.reset_optimizer: bool = False
        self.lr_policy: Optional[LRScheduler] = None

        # loss
        self.loss: Optional[nn.Module] = None

        # snapshots
        self.strict_weight_loading: bool = True

        # cudnn
        self.cudnn_benchmark: bool = True

        # augs
        self.train_augs: Optional[List[BaseAug]] = None
        self.val_augs: Optional[List[BaseAug]] = None

        # metrics
        self.train_metrics: Optional[List[BaseMetric]] = None
        self.val_metrics: Optional[List[BaseMetric]] = None

    def setup_model(self):
        raise NotImplementedError

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def train(self,
              start_snapshot_name: str = None,
              ):
        model = self.setup_model()

        train_dataset, val_dataset = self.setup_datasets()

        trainer = Trainer(batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          train_dataset=train_dataset,
                          val_dataset=val_dataset,
                          optimizer=self.optimizer,
                          loss=self.loss,
                          snapshot_dir=self.snapshot_dir,
                          logs_dir=self.logs_dir,
                          train_metrics=self.train_metrics,
                          val_metrics=self.val_metrics,
                          train_augs=self.train_augs,
                          val_augs=self.val_augs,
                          train_iters=self.train_iters,
                          show_iters=self.show_iters,
                          snapshot_iters=self.snapshot_iters,
                          )
        trainer.train(model=model,
                      reset_optimizer=self.reset_optimizer,
                      start_snapshot_name=start_snapshot_name,
                      max_iteration=self.max_iteration,
                      lr_policy=self.lr_policy,
                      strict_weight_loading=self.strict_weight_loading,
                      cudnn_benchmark=self.cudnn_benchmark,
                      )
