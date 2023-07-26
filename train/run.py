import os
from abc import ABC
from typing import Tuple, Optional, List, Type, Dict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchvision import transforms

from augmentations.classification.augs import BaseAug
from metrics.base_metric import BaseMetric
from normalize.base_normalizer import BaseNormalizer
from optim_utils.iter_policy.base_policy import BaseIterationPolicy
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
        self.train_iters: int = 300
        self.show_iters: int = 10
        self.snapshot_iters: int = 300
        self.max_iteration: int = 1000000

        self.snapshot_dir: str = os.path.join(os.environ['SNAPSHOTS_DIR'], self.project, self.experiment_name, self.run_name)
        self.logs_dir: str = os.path.join(os.environ['LOG_DIR'], self.project, self.experiment_name, self.run_name)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # optimizer
        self.optimizer_class: Optional[Type[Optimizer]] = None
        self.optimizer_kwargs: Dict = {}
        self.reset_optimizer: bool = False
        self.lr_policy: Optional[BaseIterationPolicy] = None

        # loss
        self.loss: Optional[nn.Module] = None

        # snapshots
        self.strict_weight_loading: bool = True

        # cudnn
        self.cudnn_benchmark: bool = True

        self.allow_tf32: bool = False

        # augs
        self.train_augs: Optional[List[BaseAug]] = None
        self.val_augs: Optional[List[BaseAug]] = None

        # metrics
        self.train_metrics: Optional[List[BaseMetric]] = None
        self.val_metrics: Optional[List[BaseMetric]] = None

        self.normalizer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
        self.normalize_batch: Optional[BaseNormalizer] = None

    def setup_model(self):
        raise NotImplementedError

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def train(self,
              start_snapshot_name: str = None,
              ):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

        model = self.setup_model()

        train_dataset, val_dataset = self.setup_datasets()

        trainer = Trainer(batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          train_dataset=train_dataset,
                          val_dataset=val_dataset,
                          optimizer_class=self.optimizer_class,
                          optimizer_kwargs=self.optimizer_kwargs,
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
                          normalizer=self.normalize_batch,)
        trainer.train(model=model,
                      reset_optimizer=self.reset_optimizer,
                      start_snapshot_name=start_snapshot_name,
                      max_iteration=self.max_iteration,
                      lr_policy=self.lr_policy,
                      strict_weight_loading=self.strict_weight_loading,
                      cudnn_benchmark=self.cudnn_benchmark,
                      allow_tf32=self.allow_tf32,
                      )
