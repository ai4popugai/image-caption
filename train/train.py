import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler


class Trainer:
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 batch_size,
                 num_workers,
                 snapshot_dir,
                 logs_dir,
                 optimizer,
                 loss,
                 train_iters, val_iters,
                 show_iters):
        """
        :param train_dataset: dataset for train loop.
        :param val_dataset: dataset for validation loop.
        :param batch_size: batch size for single GPU.
        :param num_workers: number of CPU workers used by DataLoader.
        :param snapshot_dir: directory for snapshots.
        :param logs_dir: directory where to place train logs.
        :param optimizer: initialized optimizer instance.
        :param loss: loss function for optimizations.
        :param train_iters: number of training iterations before validation loop is executed.
        :param val_iters: number of iterations in the validation loop.
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
        self.train_iters = train_iters
        self.val_iters = val_iters
        self.show_iters = show_iters

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, model: nn.Module, start_snapshot_path: str or None, reset_optimizer: bool,
              max_iteration: int,
              check_unused_params: bool = False,
              lr_policy: LRScheduler or None = None,
              strict_weight_loading: bool = True,
              cudnn_benchmark: bool = True,):
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32  # False to improve numerical accuracy.
        torch.backends.cudnn.allow_tf32 = self.allow_tf32  # False to improve numerical accuracy.

        if start_snapshot_path is not None:
            self._load_snapshot(model, start_snapshot_path, strict_weight_loading)

    def _load_snapshot(self, model, start_snapshot_path, strict_weight_loading):
        checkpoint = torch.load(start_snapshot_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict_weight_loading)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Loaded snapshot from {start_snapshot_path}')

    def _save_snapshot(self, model, snapshot_path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, snapshot_path)
        print(f'Saved snapshot to {snapshot_path}')

    def _train_iteration(self, model, batch):
        model.train()
        self.optimizer.zero_grad()
        result = model(batch)
        loss = self.loss(result, batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _val_iteration(self, model, batch):
        model.eval()
        with torch.no_grad():
            result = model(batch)
            loss = self.loss(result, batch)
        return loss.item()
