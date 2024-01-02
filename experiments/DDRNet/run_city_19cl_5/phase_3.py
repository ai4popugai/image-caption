"""run_city_19cl_2 continue"""

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
from transforms.segmentration import BaseToImageTransforms, FramesToImage, GroundTruthToImage, LogitsToImage


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.num_classes = 19

        self.loss = CrossEntropyLoss(result_trg_key=LOGIT_KEY, batch_trg_key=GROUND_TRUTH_KEY,
                                     ignore_index=19)

        self.train_augs = [RandomFlip(target_keys=self.target_keys),
                           RandomCrop(self.crop_size, target_keys=self.target_keys),
                           RandomResizedCropWithProb(probability=0.95,
                                                     size=[0.2, 1.0],
                                                     target_keys=self.target_keys),
                           RandomColorJitterWithProb(probability=0.8,
                                                     brightness_range=(0.7, 1.5),
                                                     contrast_range=(0.7, 1.2),
                                                     saturation_range=(0.7, 1.2)),
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
    # 28500
    Phase().train(start_snapshot=None)