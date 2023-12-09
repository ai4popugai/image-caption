from typing import Tuple

from torch import nn
from torch.utils.data import Dataset

from datasets import FRAME_KEY
from datasets.segmantation.cityscapes import CITYSCAPES_NUM_CLASSES, CityscapesDataset
from nn_models.segmentation.ddrnet.models import DDRNet23Slim
from normalize.normalize import BatchNormalizer
from train.run import Run
from torchvision import transforms


class RunBase(Run):
    def __init__(self, filename: str):
        super().__init__(filename)

        self.num_classes = CITYSCAPES_NUM_CLASSES

        self._normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalizer = BatchNormalizer(normalizer=self._normalizer, target_key=FRAME_KEY)

        self.batch_size = 64
        self.num_workers = 8

    def setup_model(self) -> nn.Module:
        return DDRNet23Slim(num_classes=self.num_classes)

    def setup_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset, val_dataset = CityscapesDataset(mode='fine', split='train'), \
                                     CityscapesDataset(mode='fine', split='val')
        return train_dataset, val_dataset
