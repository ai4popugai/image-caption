from typing import Tuple, Dict

import torch
from torch import nn
from torch.utils.data import Dataset

from augmentations.augs import RandomFlip, RandomCrop, CenterCrop, Rotate, RandomColorJitterWithProb
from datasets import FRAME_KEY, GROUND_TRUTH_KEY, LOGIT_KEY
from datasets.segmantation.cityscapes import CITYSCAPES_NUM_CLASSES, CityscapesDataset
from loss.cross_entropy import CrossEntropyLoss
from metrics.segmentation.iou import IoU
from nn_models.segmentation.ddrnet.models import DDRNet23Slim
from normalize.normalize import BatchNormalizer
from train.run import Run
from torchvision import transforms

from transforms.to_image import BaseToImageTransforms, CityscapesFramesToImage, CityscapesGroundTruthToImage, \
    CityscapesLogitsToImage


class RunBase(Run):
    def __init__(self, filename: str):
        super().__init__(filename)

        self.num_classes = CITYSCAPES_NUM_CLASSES

        self._normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalizer = BatchNormalizer(normalizer=self._normalizer, target_key=FRAME_KEY)

        self.batch_size = 4
        self.num_workers = 8

        self.train_iters = 2000
        self.batch_dump_iters = 400
        self.snapshot_iters = 2000
        self.show_iters = 20

        self.loss = CrossEntropyLoss(result_trg_key=LOGIT_KEY, batch_trg_key=GROUND_TRUTH_KEY)

        self.optimizer_class = torch.optim.Adam

        self.train_metrics = [IoU(self.num_classes)]
        self.val_metrics = [IoU(self.num_classes)]

        self.crop_size = (512, 1024)

        target_keys = [FRAME_KEY, GROUND_TRUTH_KEY]
        self.train_augs = [RandomFlip(target_keys=target_keys),
                           RandomCrop(self.crop_size, target_keys=target_keys),
                           # Rotate(angle_range=(-30, 30), target_keys=target_keys),
                           # due to rotate edges' colors become different
                           RandomColorJitterWithProb(probability=0.8,
                                                     brightness_range=(0.7, 1),
                                                     contrast_range=(0.7, 1),
                                                     # hue_range=(0.3, 0.5),
                                                     # due to hue images change their original colors
                                                     saturation_range=(0.7, 1)),

                           ]
        self.val_augs = [CenterCrop(self.crop_size, target_keys=target_keys)]

        self.batch_dump_flag = True

    def setup_model(self) -> nn.Module:
        return DDRNet23Slim(num_classes=self.num_classes)

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset, val_dataset = CityscapesDataset(mode='fine', split='train'), \
                                     CityscapesDataset(mode='fine', split='val')
        return train_dataset, val_dataset

    def get_batch_sample_to_image_map(self) -> Dict[str, BaseToImageTransforms]:
        return {FRAME_KEY: CityscapesFramesToImage(),
                GROUND_TRUTH_KEY: CityscapesGroundTruthToImage(),
                LOGIT_KEY: CityscapesLogitsToImage()}
