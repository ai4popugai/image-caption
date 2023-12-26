import os

from augmentations.augs import RandomFlip, RandomColorJitterWithProb, RandomCrop, CenterCrop
from datasets import FRAME_KEY, GROUND_TRUTH_KEY
from experiments.DDRNet.run_base import RunBase
from optim_utils.iter_policy.linear_policy import LinearIterationPolicy


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.train_iters = 1000
        self.batch_dump_iters = 1000
        self.snapshot_iters = 1000
        self.show_iters = 10

        self.crop_size = (512, 1024)

        self.target_keys = [FRAME_KEY, GROUND_TRUTH_KEY]
        self.train_augs = [RandomFlip(target_keys=self.target_keys),
                           RandomCrop(self.crop_size, target_keys=self.target_keys),
                           # Rotate(angle_range=(-30, 30), target_keys=target_keys),
                           # due to rotate edges' colors become different
                           RandomColorJitterWithProb(probability=0.8,
                                                     brightness_range=(0.7, 1),
                                                     contrast_range=(0.7, 1),
                                                     # hue_range=(0.3, 0.5),
                                                     # due to hue images change their original colors
                                                     saturation_range=(0.7, 1)),

                           ]
        self.val_augs = [CenterCrop(self.crop_size, target_keys=self.target_keys)]

        self.optimizer_kwargs = {'lr': 0., 'weight_decay': 3e-5}
        self.lr_policy = LinearIterationPolicy(start_iter=0, start_lr=0, end_iter=60000, end_lr=9e-3)


if __name__ == '__main__':
    Phase().train(start_snapshot=None)
