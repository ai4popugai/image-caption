from torch import nn

from .utils import DualResNet, BasicBlock


def DDRNet23Slim(num_classes: int) -> nn.Module:
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, planes=32, spp_planes=128, head_planes=64,
                       augment=False)
    return model


def DDRNet23(num_classes: int) -> nn.Module:
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, planes=64, spp_planes=128, head_planes=128,
                       augment=False)

    return model


def DDRNet39(num_classes: int) -> nn.Module:
    model = DualResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, planes=64, spp_planes=128, head_planes=256,
                       augment=False)
    return model

