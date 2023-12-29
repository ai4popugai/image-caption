"""
The same as run_city_19cl_0, but:
1. lr_policy and final lr have changed
2. RandomResizedCropWithProb is different to reduce over fitting
3. use more common in semantic segmentation works weight decay
"""

import os
from typing import Tuple, Dict

from torch import nn
from torch.utils.data import Dataset

from augmentations.augs import RandomFlip, RandomCrop, RandomColorJitterWithProb, RandomResizedCropWithProb
from datasets import FRAME_KEY, GROUND_TRUTH_KEY, LOGIT_KEY
from datasets.segmantation.cityscapes import CityscapesDataset19
from experiments.DDRNet.run_base import RunBase
from loss.cross_entropy import CrossEntropyLoss
from nn_models.segmentation.ddrnet.models import DDRNet23Slim
from optim_utils.iter_policy.linear_policy import LinearIterationPolicy
from transforms.segmentration import BaseToImageTransforms, FramesToImage, GroundTruthToImage, LogitsToImage


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.num_classes = 19

        self.loss = CrossEntropyLoss(result_trg_key=LOGIT_KEY, batch_trg_key=GROUND_TRUTH_KEY,
                                     ignore_index=19)

        self.optimizer_kwargs = {'lr': 0., 'weight_decay': 5e-4}
        self.lr_policy = LinearIterationPolicy(start_iter=0, start_lr=0, end_iter=20000, end_lr=1e-3)

        self.train_augs = [RandomFlip(target_keys=self.target_keys),
                           RandomCrop(self.crop_size, target_keys=self.target_keys),
                           RandomResizedCropWithProb(probability=1.,
                                                     size=[0.1, 1.0],
                                                     target_keys=self.target_keys),
                           RandomColorJitterWithProb(probability=0.8,
                                                     brightness_range=(0.7, 1),
                                                     contrast_range=(0.7, 1),
                                                     saturation_range=(0.7, 1)),
                           ]

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset, val_dataset = CityscapesDataset19(split='train'), \
                                     CityscapesDataset19(split='val')
        return train_dataset, val_dataset

    def setup_model(self) -> nn.Module:
        return DDRNet23Slim(num_classes=self.num_classes)

    def get_batch_sample_to_image_map(self) -> Dict[str, BaseToImageTransforms]:
        return {FRAME_KEY: FramesToImage(),
                GROUND_TRUTH_KEY: GroundTruthToImage(color_map=CityscapesDataset19.color_map),
                LOGIT_KEY: LogitsToImage(color_map=CityscapesDataset19.color_map)}


if __name__ == '__main__':
    Phase().train(start_snapshot=None)
