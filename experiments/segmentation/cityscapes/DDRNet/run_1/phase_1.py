import os

from augmentations.augs import RandomFlip, RandomCrop, RandomResizedCropWithProb, RandomColorJitterWithProb
from experiments.segmentation.cityscapes.DDRNet.run_base import RunBase


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'lr': 5e-4, 'weight_decay': 5e-4}

        self.train_augs = [RandomFlip(target_keys=self.target_keys),
                           RandomCrop(self.crop_size, target_keys=self.target_keys),
                           RandomResizedCropWithProb(probability=0.95,
                                                     size=[0.5, 1.0],
                                                     target_keys=self.target_keys),
                           RandomColorJitterWithProb(probability=0.95,
                                                     brightness_range=(0.7, 1.3),
                                                     contrast_range=(0.7, 1.2),
                                                     saturation_range=(0.7, 1.2)),
                           ]


if __name__ == '__main__':
    start_snapshot = 'DDRNet/run_city_0/DDRNet23Slim_60000.pth'
    Phase().train(start_snapshot=start_snapshot)
