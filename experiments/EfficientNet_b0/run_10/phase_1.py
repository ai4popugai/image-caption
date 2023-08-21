import os

from augmentations.classification.augs import RandomFlip, RandomCrop, Rotate
from experiments.EfficientNet_b0.run_base import RunBase


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'lr': 3e-4, 'weight_decay': 3e-3}
        self.lr_policy = None

        self.train_augs = [RandomFlip(), Rotate(), RandomCrop(self.crop_size)]


if __name__ == '__main__':
    Phase().train(start_snapshot=None)
