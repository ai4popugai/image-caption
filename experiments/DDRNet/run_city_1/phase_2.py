import os

from augmentations.augs import RandomFlip, RandomCrop, RandomResizedCropWithProb, RandomColorJitterWithProb
from experiments.DDRNet.run_base import RunBase


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'lr': 5e-4, 'weight_decay': 5e-4}

        self.train_augs = [RandomFlip(target_keys=self.target_keys),
                           RandomCrop(self.crop_size, target_keys=self.target_keys),
                           RandomResizedCropWithProb(probability=0.95,
                                                     size=[1.1, 2.0],
                                                     target_keys=self.target_keys,
                                                     inpaint_val=self.ignore_index
                                                     ),
                           RandomColorJitterWithProb(probability=0.95,
                                                     brightness_range=(0.7, 1.3),
                                                     contrast_range=(0.7, 1.2),
                                                     saturation_range=(0.7, 1.2)),
                           ]


if __name__ == '__main__':
    # 106000
    Phase().train(start_snapshot=None)
