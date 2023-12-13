from typing import Dict, Optional

import torch
from torch import nn

from datasets import LOGIT_KEY, FRAME_KEY
from .utils import DualResNet, BasicBlock


class BaseDDRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model: Optional[DualResNet] = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {LOGIT_KEY: self.model(batch[FRAME_KEY])}


class DDRNet23Slim(BaseDDRNet):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.num_classes, planes=32, spp_planes=128,
                                head_planes=64, augment=False)


class DDRNet23(BaseDDRNet):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.num_classes, planes=64, spp_planes=128,
                                head_planes=128, augment=False)


class DDRNet39(BaseDDRNet):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.model = DualResNet(BasicBlock, [3, 4, 6, 3], num_classes=self.num_classes, planes=64, spp_planes=128,
                                head_planes=256, augment=False)
