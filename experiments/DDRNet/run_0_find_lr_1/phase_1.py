import os

from augmentations.augs import RandomFlip, RandomCrop, RandomColorJitterWithProb, CenterCrop
from experiments.DDRNet.run_base import RunBase
from optim_utils.iter_policy.linear_policy import LinearIterationPolicy


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'lr': 0., 'weight_decay': 3e-5}
        self.lr_policy = LinearIterationPolicy(start_iter=0, start_lr=0, end_lr=4000, end_val=1e-2)


if __name__ == '__main__':
    Phase().train(start_snapshot=None)
