import os
from typing import Tuple, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, random_split

from augmentations.augs import RandomFlip, CenterCrop, RandomColorJitterWithProb
from datasets import FRAME_KEY, GROUND_TRUTH_KEY, LOGIT_KEY
from datasets.segmantation.dubai_aerial import DubaiAerial
from loss.cross_entropy import CrossEntropyLoss
from metrics.segmentation.iou import IoU
from nn_models.segmentation.unet import Unet
from normalize.normalize import BatchNormalizer
from train.run import Run
from torchvision import transforms

from transforms.segmentration import FramesToImage, GroundTruthToImage, LogitsToImage, BaseToImageTransforms

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


class RunBase(Run):
    def __init__(self, filename: str):
        super().__init__(filename)

        self.num_classes = 6

        self._normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalizer = BatchNormalizer(normalizer=self._normalizer, target_key=FRAME_KEY)

        self.batch_size = 32
        self.num_workers = 0

        self.train_iters = 50
        self.batch_dump_iters = 50
        self.snapshot_iters = 50
        self.show_iters = 1

        self.loss = CrossEntropyLoss(result_trg_key=LOGIT_KEY, batch_trg_key=GROUND_TRUTH_KEY)

        self.optimizer_class = torch.optim.Adam

        self.train_metrics = [IoU(self.num_classes)]
        self.val_metrics = [IoU(self.num_classes)]

        self.resolution = (256, 256)

        self.target_keys = [FRAME_KEY, GROUND_TRUTH_KEY]

        self.train_augs = [RandomFlip(target_keys=self.target_keys),
                           RandomColorJitterWithProb(probability=0.8,
                                                     brightness_range=(0.7, 1.3),
                                                     contrast_range=(0.7, 1.2),
                                                     saturation_range=(0.7, 1.2)),
                           ]

        self.batch_dump_flag = True

    def setup_model(self) -> nn.Module:
        return Unet(num_classes=self.num_classes, encoder_weights='imagenet')

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        torch.manual_seed(42)
        dataset = DubaiAerial(self.resolution)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset

    def get_batch_sample_to_image_map(self) -> Dict[str, BaseToImageTransforms]:
        return {FRAME_KEY: FramesToImage(),
                GROUND_TRUTH_KEY: GroundTruthToImage(color_map=DubaiAerial.color_map),
                LOGIT_KEY: LogitsToImage(color_map=DubaiAerial.color_map)}
