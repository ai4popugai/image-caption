import os
from typing import List, Dict, Union, Tuple, Any, Iterator

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from augmentations.classification.augs import BaseAug
from metricks.base_metric import BaseMetric


class Trainer:
    def __init__(self,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 batch_size: int,
                 num_workers: int,
                 snapshot_dir: str,
                 logs_dir: str,
                 optimizer: torch.optim.Optimizer,
                 loss: nn.Module,
                 train_metrics: List[BaseMetric],
                 val_metrics: List[BaseMetric],
                 train_augs: List[BaseAug],
                 val_augs: List[BaseAug],
                 train_iters: int, snapshot_iters: int,
                 show_iters: int,):
        """
        :param train_dataset: dataset for train loop.
        :param val_dataset: dataset for validation loop.
        :param batch_size: batch size for single GPU.
        :param num_workers: number of CPU workers used by DataLoader.
        :param snapshot_dir: directory for snapshots.
        :param logs_dir: directory where to place train logs.
        :param optimizer: initialized optimizer instance.
        :param loss: loss function for optimizations.
        :param train_metrics: list of metrics to be calculated during training.
        :param val_metrics: list of metrics to be calculated during validation.
        :param train_augs: list of augmentations to be applied during training.
        :param val_augs: list of augmentations to be applied during validation.
        :param train_iters: number of training iterations before log train loss and training metrics.
        :param snapshot_iters: number of training iterations before snapshot is saved.
        :param show_iters: number of training iterations to accumulate loss for logging to tensorboard.
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.snapshot_dir = snapshot_dir
        self.logs_dir = logs_dir
        self.optimizer = optimizer
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.train_augs = train_augs
        self.val_augs = val_augs
        self.snapshot_iters = snapshot_iters
        self.train_iters = train_iters
        self.show_iters = show_iters

        if os.path.exists(self.logs_dir) is False:
            os.makedirs(self.logs_dir)
        writer = SummaryWriter(self.logs_dir)
        self.writer = writer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, model: nn.Module, start_snapshot_path: str or None, reset_optimizer: bool,
              max_iteration: int,
              lr_policy: LRScheduler or None = None,
              strict_weight_loading: bool = True,
              cudnn_benchmark: bool = True,):

        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32  # False to improve numerical accuracy.
        torch.backends.cudnn.allow_tf32 = self.allow_tf32  # False to improve numerical accuracy.

        if start_snapshot_path is not None:
            global_step = self._load_snapshot(model, start_snapshot_path, strict_weight_loading)
        else:
            the_last_snapshot = self._detect_last_snapshot()
            if the_last_snapshot is not None:
                global_step = self._load_snapshot(model, the_last_snapshot, strict_weight_loading)
            else:
                global_step = 0

        if reset_optimizer:
            self.optimizer.zero_grad()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        self._train_loop(model, train_loader, val_loader, global_step, max_iteration, lr_policy)

    def _load_snapshot(self, model, start_snapshot_path, strict_weight_loading) -> int:
        checkpoint = torch.load(start_snapshot_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict_weight_loading)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Loaded snapshot from {start_snapshot_path}')
        return checkpoint['global_step']

    def _detect_last_snapshot(self) -> str or None:
        snapshot_paths = [os.path.join(self.snapshot_dir, name) for name in os.listdir(self.snapshot_dir)]
        if len(snapshot_paths) == 0:
            return None
        return max(snapshot_paths, key=os.path.getctime)

    def _save_snapshot(self, model, snapshot_path, global_step):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': global_step,

        }, snapshot_path)
        print(f'Saved snapshot to {snapshot_path}')

    @staticmethod
    def _update_metrics(metrics: List[BaseMetric],
                        result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        for metric in metrics:
            metric.update(result, batch)

    def _report_metrics(self, mode: str,  metrics: List[BaseMetric], global_step) -> None:
        if mode != 'train' and mode != 'val':
            raise ValueError(f'Unknown mode: {mode}')

        for metric in metrics:
            metric_name = metric.__class__.__name__
            metric_value = metric.compute()
            metric.reset()
            self.writer.add_scalars(metric_name, {mode: metric_value}, global_step)
            print(f'Iteration: {global_step}, {mode} {metric_name}: {metric_value}')

    def _train_iteration(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> float:
        model.train()
        self.optimizer.zero_grad()
        result = model(batch)
        loss = self.loss(result, batch)
        loss.backward()
        self.optimizer.step()
        self._update_metrics(self.train_metrics, result, batch)
        return loss.item()

    def _val_iteration(self, model, batch) -> torch.Tensor:
        result = model(batch)
        loss = self.loss(result, batch)
        self._update_metrics(self.val_metrics, result, batch)
        return loss.item()

    def _aug_loop(self, aug_list: List[BaseAug], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for aug in aug_list:
            batch = aug(batch)
        return batch

    @staticmethod
    def _get_batch(iterator, train_loader) -> Union[Iterator, Dict[str, torch.Tensor]]:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        return iterator, batch

    def _train_loop(self, model, train_loader, val_loader, start_iteration, max_iteration, lr_policy):
        iterator = iter(train_loader)
        for iteration in range(start_iteration, max_iteration + start_iteration):
            iterator, batch = self._get_batch(iterator, train_loader)
            batch = self._aug_loop(self.train_augs, batch)
            loss = self._train_iteration(model, batch)
            iteration += 1
            if iteration % self.snapshot_iters == 0:
                # save snapshot
                self._save_snapshot(model, f'{self.snapshot_dir}/snapshot_{iteration}.pth', iteration)
            if iteration % self.show_iters == 0:
                # report loss
                print(f'Iteration: {iteration}, train loss: {loss}')
                self.writer.add_scalars("Loss", {'train': loss}, iteration)

                # report metrics
                self._report_metrics('train', self.train_metrics, iteration)
            if iteration % self.train_iters == 0:
                loss = self._val_loop(model, val_loader)

                # report loss
                self.writer.add_scalars("Loss", {'val': loss}, iteration)
                print(f'Validation iteration: {iteration}, mean loss: {loss}')

                # report metrics
                self._report_metrics('val', self.val_metrics, iteration)
            if iteration == max_iteration:
                break
            if lr_policy is not None:
                lr_policy.step()

    def _val_loop(self, model, val_loader) -> float:
        losses = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = self._aug_loop(self.val_augs, batch)
                loss = self._val_iteration(model, batch)
                losses.append(loss)
        mean_loss = sum(losses) / len(losses)
        return mean_loss.item()
