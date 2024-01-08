import os
from typing import Tuple, Dict

import torch
from torch import nn
from torch.utils.data import Dataset

from augmentations.augs import RandomFlip, RandomCrop, CenterCrop, RandomColorJitterWithProb, RandomResizedCropWithProb
from datasets import FRAME_KEY, GROUND_TRUTH_KEY, LOGIT_KEY
from datasets.segmantation.cityscapes import CityscapesDataset19
from loss.cross_entropy import CrossEntropyLoss
from metrics.segmentation.iou import IoU
from nn_models.segmentation.ddrnet.models import DDRNet23Slim
from normalize.normalize import BatchNormalizer
from train.run import Run
from torchvision import transforms

from transforms.segmentration import FramesToImage, GroundTruthToImage, LogitsToImage, BaseToImageTransforms

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


class RunBase(Run):
    def __init__(self, filename: str):
        super().__init__(filename)

        self.num_classes = 19

        self._normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalizer = BatchNormalizer(normalizer=self._normalizer, target_key=FRAME_KEY)

        self.batch_size = 8
        self.num_workers = 4

        self.train_iters = 500
        self.batch_dump_iters = 500
        self.snapshot_iters = 500
        self.show_iters = 5
        self.accum_iters = 4

        self.loss = CrossEntropyLoss(result_trg_key=LOGIT_KEY, batch_trg_key=GROUND_TRUTH_KEY,
                                     ignore_index=19)

        self.optimizer_class = torch.optim.Adam

        self.train_metrics = [IoU(self.num_classes)]
        self.val_metrics = [IoU(self.num_classes)]

        self.crop_size = (1024, 1024)

        self.target_keys = [FRAME_KEY, GROUND_TRUTH_KEY]

        self.train_augs = [RandomFlip(target_keys=self.target_keys),
                           RandomCrop(self.crop_size, target_keys=self.target_keys),
                           RandomResizedCropWithProb(probability=0.95,
                                                     size=[0.2, 1.0],
                                                     target_keys=self.target_keys),
                           RandomColorJitterWithProb(probability=0.95,
                                                     brightness_range=(0.7, 1.3),
                                                     contrast_range=(0.7, 1.2),
                                                     saturation_range=(0.7, 1.2)),
                           ]
        self.val_augs = [CenterCrop(self.crop_size, target_keys=self.target_keys)]

        self.batch_dump_flag = True

    def setup_model(self) -> nn.Module:
        return DDRNet23Slim(num_classes=self.num_classes)

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset, val_dataset = CityscapesDataset19(mode='fine', split='train'), \
                                     CityscapesDataset19(mode='fine', split='val')
        return train_dataset, val_dataset

    def get_batch_sample_to_image_map(self) -> Dict[str, BaseToImageTransforms]:
        return {FRAME_KEY: FramesToImage(),
                GROUND_TRUTH_KEY: GroundTruthToImage(color_map=CityscapesDataset19.color_map),
                LOGIT_KEY: LogitsToImage(color_map=CityscapesDataset19.color_map)}
