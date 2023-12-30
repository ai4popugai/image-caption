import datetime
import os
from typing import List, Dict, Union, Iterator, Optional, Type, Callable

import cv2
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from augmentations.augs import BaseAug
from metrics.base_metric import BaseMetric
from optim_utils.iter_policy.base_policy import BaseIterationPolicy
from optim_utils.iter_policy.policy import LrPolicy
from transforms.segmentration import BaseToImageTransforms

MODEL_STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
GLOBAL_STEP_KEY = 'global_step'
TRAIN_MODE = 'train'
VAL_MODE = 'val'


class Trainer:
    def __init__(self,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 batch_size: int,
                 num_workers: int,
                 snapshot_dir: str,
                 logs_dir: str,
                 batch_dump_dir: str,
                 optimizer_class: Type[Optimizer],
                 optimizer_kwargs: Dict,
                 loss: nn.Module,
                 train_iters: int, snapshot_iters: int,
                 batch_dump_iters: int,
                 show_iters: int,
                 normalizer: Optional[Callable] = None,
                 train_metrics: Optional[List[BaseMetric]] = None,
                 val_metrics: Optional[List[BaseMetric]] = None,
                 train_augs: Optional[List[BaseAug]] = None,
                 val_augs: Optional[List[BaseAug]] = None,
                 device: Optional[Union[torch.device, str]] = None,
                 batch_dump_flag: bool = False,
                 sample_to_image: Optional[Dict[str, BaseToImageTransforms]] = None
                 ):
        """
        :param train_dataset: dataset for train loop.
        :param val_dataset: dataset for validation loop.
        :param batch_size: batch size for single GPU.
        :param num_workers: number of CPU workers used by DataLoader.
        :param snapshot_dir: directory for snapshots.
        :param logs_dir: directory where to place train logs.
        :param batch_dump_dir: directory to batch_dump media files.
        :param optimizer_class: optimizer class to be initialized.
        :param optimizer_kwargs: kwargs for optimizer initialization.
        :param loss: loss function for optimizations.
        :param train_metrics: list of metrics to be calculated during training.
        :param val_metrics: list of metrics to be calculated during validation.
        :param train_augs: list of augmentations to be applied during training.
        :param val_augs: list of augmentations to be applied during validation.
        :param train_iters: number of training iterations before log train loss and training metrics.
        :param snapshot_iters: number of training iterations before snapshot is saved.
        :param batch_dump_iters: number of training iterations before batch_dump batch.
        :param show_iters: number of training iterations to accumulate loss for logging to tensorboard.
        :param normalizer: normalization layer to be applied to input images.
        :param device: device to train on.
        :param: batch_dump_flag: True to batch batch_dump. Default False
        :param sample_to_image: map to convert batch sample(value) by key to image.
        Must be not None if batch_dump_flag is True
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.snapshot_dir = snapshot_dir
        self.logs_dir = logs_dir
        self.batch_dump_dir = batch_dump_dir
        self.optimizer = None
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.train_augs = train_augs
        self.val_augs = val_augs
        self.snapshot_iters = snapshot_iters
        self.batch_dump_iters = batch_dump_iters
        self.train_iters = train_iters
        self.show_iters = show_iters
        self.normalizer = normalizer

        if os.path.exists(self.logs_dir) is False:
            os.makedirs(self.logs_dir)

        # setup device
        self.device = self.get_device() if device is None \
            else torch.device(device) if isinstance(device, str) else device
        print(f'train on {self.device}')

        # batch_dump
        if batch_dump_flag is True and sample_to_image is None:
            raise RuntimeError('sample_to_image should be set up if batch_dump_flag is True.')
        self.batch_dump_flag = batch_dump_flag
        self.sample_to_image = sample_to_image

    @staticmethod
    def get_device() -> torch.device:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def train(self, model: nn.Module, start_snapshot: str or None, reset_optimizer: bool,
              max_iteration: int,
              lr_policy: Optional[BaseIterationPolicy] = None,
              strict_weight_loading: bool = True,
              cudnn_benchmark: bool = True,
              allow_tf32: bool = False,):

        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # False to improve numerical accuracy.
        torch.backends.cudnn.allow_tf32 = allow_tf32  # False to improve numerical accuracy.

        # move model to device
        model.to(self.device)

        # do not move metrics to device
        # all metrics by default must be computed and stored on cpu
        # if self.train_metrics is not None:
        #     for metric in self.train_metrics:
        #         metric.to(self.device)
        #
        # if self.val_metrics is not None:
        #     for metric in self.val_metrics:
        #         metric.to(self.device)

        # instantiating optimizer
        self.optimizer = self.optimizer_class(model.parameters(), **self.optimizer_kwargs)

        # init lr_policy
        if lr_policy is not None:
            lr_policy = LrPolicy(self.optimizer, lr_policy)

        # load snapshot
        the_last_snapshot_path = self._detect_last_snapshot_path()
        if the_last_snapshot_path is not None:
            global_step = self._load_snapshot(model, the_last_snapshot_path, strict_weight_loading, reset_optimizer)
        else:
            if start_snapshot is not None:
                global_step = self._load_snapshot(model, start_snapshot, strict_weight_loading, reset_optimizer)
            else:
                global_step = 0

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        self._train_loop(model, train_loader, val_loader, global_step, max_iteration, lr_policy)

    def _load_snapshot(self, model: nn.Module, snapshot_path: str, strict_weight_loading: bool,
                       reset_optimizer: bool) -> int:
        checkpoint = torch.load(snapshot_path, map_location=self.device)
        model.load_state_dict(checkpoint[MODEL_STATE_DICT_KEY], strict=strict_weight_loading)
        if reset_optimizer is False:
            if OPTIMIZER_STATE_DICT_KEY in checkpoint:
                self.optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_DICT_KEY])
            else:
                print('No optimizer state dict in snapshot. Optimizer is reset.')
        print(f'Loaded snapshot from {snapshot_path}')
        return checkpoint[GLOBAL_STEP_KEY]

    def _detect_last_snapshot_path(self) -> str or None:
        snapshot_paths = [os.path.join(self.snapshot_dir, name) for name in sorted(os.listdir(self.snapshot_dir))]
        if len(snapshot_paths) == 0:
            return None
        return max(snapshot_paths, key=os.path.getctime)

    def _save_snapshot(self, model: nn.Module, snapshot_path: str, global_step: int):
        torch.save({
            MODEL_STATE_DICT_KEY: model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
            GLOBAL_STEP_KEY: global_step,

        }, snapshot_path)
        print(f'Saving snapshot to {snapshot_path}')

    @staticmethod
    def _update_metrics(metrics: List[BaseMetric],
                        result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        for metric in metrics:
            metric.update(result, batch)

    def _report_metrics(self, mode: str,  metrics: Optional[List[BaseMetric]], global_step: int,
                        log_msg: str) -> str:
        if metrics is not None:
            for metric in metrics:
                metric_name = metric.name
                metric_value = metric.compute()
                metric.reset()
                with SummaryWriter(self.logs_dir) as w_hp:
                    w_hp.add_scalars(metric_name, {mode: metric_value}, global_step)
                log_msg += f',{" val" if mode == "val" else ""} {metric_name}: {metric_value:.2f}{metric.unit}'
        return log_msg

    def _batch_dump(self, batch: Dict[str, torch.Tensor], iteration: int, mode: str,):
        if iteration % self.batch_dump_iters == 0 and self.batch_dump_flag:
            for key in batch:
                imgs = self.sample_to_image[key](batch[key])
                for i, img in enumerate(imgs):
                    cv2.imwrite(os.path.join(self.batch_dump_dir, f'{mode}_iter_{iteration}__{i}_{key}.png'), img)

    def _train_iteration(self, model: nn.Module, batch: Dict[str, torch.Tensor], iteration: int) -> float:
        model.train()
        self.optimizer.zero_grad()
        result = model(batch)
        self._batch_dump(result, iteration, TRAIN_MODE)
        loss = self.loss(result, batch)
        loss.backward()
        self.optimizer.step()
        self._update_metrics(self.train_metrics, result, batch)
        return loss.item()

    def _val_iteration(self, model: nn.Module, batch: Dict[str, torch.Tensor], global_iter: int) -> float:
        result = model(batch)
        self._batch_dump(result, global_iter, mode=VAL_MODE)
        loss = self.loss(result, batch)
        self._update_metrics(self.val_metrics, result, batch)
        return loss.item()

    @staticmethod
    def aug_loop(batch: Dict[str, torch.Tensor], aug_list: Optional[List[BaseAug]]) -> Dict[str, torch.Tensor]:
        if aug_list is not None:
            for aug in aug_list:
                batch = aug(batch)
        return batch

    @staticmethod
    def normalize(batch: Dict[str, torch.Tensor], normalizer: Optional[Callable]) -> Dict[str, torch.Tensor]:
        if normalizer is not None:
            batch = normalizer(batch)
        return batch
    
    @staticmethod
    def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        batch = {key: batch[key].to(device) for key in batch}
        return batch

    @staticmethod
    def _get_batch(iterator: Iterator, loader: DataLoader) -> Union[Iterator, Dict[str, torch.Tensor]]:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        return iterator, batch

    def _train_loop(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    start_iteration: int, max_iteration: int, lr_policy: Optional[LrPolicy]) -> None:
        iterator = iter(train_loader)
        lr = self.optimizer.param_groups[0]['lr']
        for iteration in range(start_iteration, max_iteration + start_iteration):
            iterator, batch = self._get_batch(iterator, train_loader)
            batch = self.aug_loop(batch, self.train_augs)
            batch = self.batch_to_device(batch, self.device)
            self._batch_dump(batch, iteration, mode=TRAIN_MODE)
            batch = self.normalize(batch, self.normalizer)
            loss = self._train_iteration(model, batch, iteration)
            iteration += 1
            if str(self.device) == 'mps':
                torch.mps.empty_cache()

            if iteration % self.show_iters == 0:
                # report loss
                log_msg = f'iter: {iteration}, loss: {loss:.3f}, lr: {lr:.6f}'
                with SummaryWriter(self.logs_dir) as w_hp:
                    w_hp.add_scalars("Loss", {TRAIN_MODE: loss}, iteration)
                    w_hp.add_scalar("lr", lr, iteration)

                # report metrics
                log_msg = self._report_metrics(TRAIN_MODE, self.train_metrics, iteration, log_msg)
                log_msg += f'{7*" "}{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                print(log_msg)
            if iteration % self.snapshot_iters == 0:
                # save snapshot
                self._save_snapshot(model, f'{self.snapshot_dir}/{model.__class__.__name__}_{iteration}.pth', iteration)
            if iteration % self.train_iters == 0:
                loss = self._val_loop(model, val_loader, iteration)

                # report loss
                log_msg = f'iter: {iteration}, val loss: {loss:.3f}'
                with SummaryWriter(self.logs_dir) as w_hp:
                    w_hp.add_scalars("Loss", {VAL_MODE: loss}, iteration)

                # report metrics
                log_msg = self._report_metrics(VAL_MODE, self.val_metrics, iteration, log_msg)
                log_msg += f'{7*" "}{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                print(log_msg)
            if lr_policy is not None:
                lr = lr_policy.step(iteration)
            if iteration == max_iteration:
                break

    def _val_loop(self, model: nn.Module, val_loader: DataLoader, global_iter: int) -> float:
        losses = []
        model.eval()
        iterator = iter(val_loader)
        val_iters = len(val_loader)
        with torch.inference_mode():
            for _ in range(val_iters):
                iterator, batch = self._get_batch(iterator, val_loader)
                batch = self.aug_loop(batch, self.val_augs)
                batch = self.batch_to_device(batch, self.device)
                self._batch_dump(batch, global_iter, mode=VAL_MODE)
                batch = self.normalize(batch, self.normalizer)
                loss = self._val_iteration(model, batch, global_iter)
                losses.append(loss)
        mean_loss = sum(losses) / len(losses)
        return mean_loss
