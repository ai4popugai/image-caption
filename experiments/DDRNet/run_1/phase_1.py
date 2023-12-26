import os

from augmentations.augs import RandomFlip, RandomCrop, RandomColorJitterWithProb, RandomResizedCropWithProb
from experiments.DDRNet.run_base import RunBase
from optim_utils.iter_policy.linear_policy import LinearIterationPolicy


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'lr': 0., 'weight_decay': 3e-5}
        self.lr_policy = LinearIterationPolicy(start_iter=0, start_lr=0, end_iter=10000, end_lr=2.5e-2)

        self.train_augs = [RandomFlip(target_keys=self.target_keys),
                           RandomCrop(self.crop_size, target_keys=self.target_keys),
                           RandomResizedCropWithProb(probability=0.8,
                                                     size=[0.25, 0.1],
                                                     target_keys=self.target_keys),
                           RandomColorJitterWithProb(probability=0.8,
                                                     brightness_range=(0.7, 1),
                                                     contrast_range=(0.7, 1),
                                                     saturation_range=(0.7, 1)),
                           ]


if __name__ == '__main__':
    Phase().train(start_snapshot=None)
