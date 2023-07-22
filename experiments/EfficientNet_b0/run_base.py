from typing import Tuple

from torch.utils.data import Dataset


class RunBase:
    def __init__(self, filename: str):


        self.batch_size = 64
        self.num_workers = 8

        self.snapshot_dir = None
        self.logs_dir = None

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

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        pass

    def train(self,
              reset_optimizer: bool = False,
              start_snapshot: str = None,
              ):
        pass