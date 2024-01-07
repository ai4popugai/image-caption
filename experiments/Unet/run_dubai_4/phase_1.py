import os

from augmentations.augs import RandomFlip, RandomColorJitterWithProb
from datasets import LOGIT_KEY, GROUND_TRUTH_KEY
from experiments.Unet.run_base import RunBase
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'lr': 5e-4, 'weight_decay': 5e-4}

        self.loss = [DiceLoss(result_trg_key=LOGIT_KEY, batch_trg_key=GROUND_TRUTH_KEY),
                     FocalLoss(result_trg_key=LOGIT_KEY, batch_trg_key=GROUND_TRUTH_KEY)]

        self.train_augs = [RandomFlip(target_keys=self.target_keys),
                           RandomColorJitterWithProb(probability=0.95,
                                                     brightness_range=(0.7, 1.3),
                                                     contrast_range=(0.7, 1.2),
                                                     saturation_range=(0.7, 1.2)),
                           ]


if __name__ == '__main__':
    Phase().train(start_snapshot=None)
