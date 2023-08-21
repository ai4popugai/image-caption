import os

from augmentations.classification.augs import RandomFlip, RandomCrop
from experiments.EfficientNet_b0.run_base import RunBase


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'lr': 2.4e-3, 'weight_decay': 3e-5}
        self.lr_policy = None


if __name__ == '__main__':
    Phase().train(start_snapshot=None)
