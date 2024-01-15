import math
import random
from abc import abstractmethod, ABC
from typing import Dict, Tuple, Union, List, Optional

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import hflip
import torchvision.transforms.functional as F
from datasets import FRAME_KEY

torchvision.disable_beta_transforms_warning()
from torchvision.transforms.v2.functional import crop


def check_batch(batch: Dict[str, torch.Tensor], target_keys: List[str]):
    if len(target_keys) > 1 and all(batch[target_keys[0]].shape[-2:] == batch[target_keys[i]].shape[-2:] for i in
                                    range(1, len(target_keys))) is False:
        raise RuntimeError("Can't augment to due dimension inequality, augment separately instead.")


class BaseAug(ABC, nn.Module):
    def __init__(self, target_keys: Optional[List[str]] = None):
        self.target_keys = target_keys if target_keys is not None else [FRAME_KEY]
        super().__init__()

    @abstractmethod
    def forward(self, frames: torch.Tensor):
        pass


class RandomFlip(BaseAug):
    def __init__(self, p=0.5, target_keys: Optional[List[str]] = None):
        super().__init__(target_keys)
        self.p = p

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply horizontal flip with probability to every target data in batch.

        :param batch: batch with target keys to apply augmentation.
        :return: batch
        """
        if torch.rand(1).item() < self.p:
            for key in self.target_keys:
                batch[key] = hflip(batch[key])

        return batch


class RandomCrop(BaseAug):
    def __init__(self, size: Tuple[int, int], target_keys: Optional[List[str]] = None):
        super().__init__(target_keys)
        self.size = size

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply the same crop to every target data in batch.

        :param batch: batch with target keys to apply augmentation.
        :return: batch
        """
        check_batch(batch, self.target_keys)
        i, j, h, w = transforms.RandomCrop.get_params(batch[self.target_keys[0]],
                                                      output_size=self.size)
        for key in self.target_keys:
            batch[key] = crop(batch[key], i, j, h, w)

        return batch


class RandomResizedCropWithProb(BaseAug):
    def __init__(self, size: Union[List[float], Tuple[int, int]],
                 probability: float = 0.5, target_keys: Optional[List[str]] = None,
                 inpaint_val: Optional[int] = None):
        """

        :param size: scaling interval or certain resolution.
        :param probability: prob of applying aug (for every instance of set of target keys)
        :param target_keys: target keys to apply aug with the same parameters.
        :param inpaint_val: value to image inpainting if change factor > 1.
        """
        if isinstance(size, List) and size[1] > 1 and inpaint_val is None:
            raise RuntimeError('Input value must be set up if change factor > 1.')
        super().__init__(target_keys)
        self.size = size
        self.probability = probability
        self.inpaint_val = inpaint_val

    def _inpaint(self, target: torch.Tensor,
                 left_margin: int, right_margin: int,
                 top_margin: int, bottom_margin: int,
                 device: torch.device) -> torch.Tensor:
        # top
        fill_shape = list(target.shape)
        fill_shape[-2] = top_margin 
        target = torch.cat((torch.full(fill_shape,
                                       self.inpaint_val, device=device),
                            target),
                           dim=-2)
        # bottom
        fill_shape = list(target.shape)
        fill_shape[-2] = bottom_margin 
        target = torch.cat((target,
                            torch.full(fill_shape,
                                       self.inpaint_val, device=device)),
                           dim=-2)
        # left
        fill_shape = list(target.shape)
        fill_shape[-1] = left_margin
        target = torch.cat((torch.full(fill_shape,
                                       self.inpaint_val, device=device),
                            target),
                           dim=-1)
        # right
        fill_shape = list(target.shape)
        fill_shape[-1] = right_margin
        target = torch.cat((target,
                            torch.full(fill_shape,
                                       self.inpaint_val, device=device)),
                           dim=-1)
        return target

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply the same crop with probability to every target data in batch.

        :param batch: batch with target keys to apply augmentation.
        :return: batch
        """
        check_batch(batch, self.target_keys)
        orig_resolution = batch[self.target_keys[0]].shape[-2:]
        device = batch[self.target_keys[0]].device
        is_inpaint = False
        left_margin = None
        right_margin = None
        top_margin = None
        bottom_margin = None
        x, y, h, w = None, None, None, None

        # Perform random resized crop on each frame
        for i in range(batch[self.target_keys[0]].shape[0]):
            if random.random() < self.probability:
                if isinstance(self.size, List):
                    change_factor = math.sqrt(random.uniform(self.size[0], self.size[1]))
                    if change_factor > 1:
                        change_factor = 1 / change_factor
                        is_inpaint = True
                    new_height = int(orig_resolution[0] * change_factor)
                    new_width = int(orig_resolution[1] * change_factor)

                    size = (new_height, new_width)
                else:
                    size = self.size

                if is_inpaint:
                    top_margin = random.randint(0, orig_resolution[0] - size[0])
                    bottom_margin = orig_resolution[0] - size[0] - top_margin
                    left_margin = random.randint(0, orig_resolution[1] - size[1])
                    right_margin = orig_resolution[1] - size[1] - left_margin
                    resize = transforms.Resize(size,
                                               antialias=False, interpolation=F.InterpolationMode.NEAREST)
                else:
                    x, y, h, w = transforms.RandomCrop.get_params(batch[self.target_keys[0]],
                                                                  output_size=size)
                    resize = transforms.Resize(orig_resolution,
                                               antialias=False, interpolation=F.InterpolationMode.NEAREST)
                for key in self.target_keys:
                    if is_inpaint:
                        # resize
                        trg = resize(batch[key][i].unsqueeze(dim=0)).squeeze(dim=0)
                        batch[key][i] = self._inpaint(trg,
                                                      left_margin, right_margin, top_margin, bottom_margin,
                                                      device=device)
                    else:
                        batch[key][i] = resize(crop(batch[key][i].unsqueeze(dim=0), x, y, h, w)).squeeze(dim=0)

        return batch


class CenterCrop(BaseAug):
    def __init__(self, size: Tuple[int, int], target_keys: Optional[List[str]] = None):
        super().__init__(target_keys)
        self.size = size

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply center crop to every target data in batch.

        :param batch: batch with target keys to apply augmentation.
        :return: batch
        """
        check_batch(batch, self.target_keys)
        for key in self.target_keys:
            batch[key] = transforms.CenterCrop(self.size)(batch[key])

        return batch


class Rotate(BaseAug):
    def __init__(self, angle_range=(-180, 180), target_keys: Optional[List[str]] = None):
        super().__init__(target_keys)
        self.angle_range = angle_range

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply rotation to every target data in batch.

        :param batch: batch with target keys to apply augmentation.
        :return: batch
        """
        angle = torch.FloatTensor(1).uniform_(self.angle_range[0], self.angle_range[1]).item()
        for key in self.target_keys:
            batch[key] = self.rotate_frames(batch[key], angle)

        return batch

    @staticmethod
    def rotate_frames(frame_batch, angle):
        """
        Method to rotate batch of frames.

        :param frame_batch: batch of frames
        :param angle: angle to rotate
        :return:
        """
        # Calculate image center
        center = torch.tensor(frame_batch.shape[-2:]).float() / 2.0

        # Perform rotation
        rotated_frames = F.rotate(frame_batch, angle, interpolation=F.InterpolationMode.BILINEAR,
                                  center=center.tolist())

        return rotated_frames


class RotateWithProb(BaseAug):
    def __init__(self, angle_range=(-180, 180), probability: float = 0.5, target_keys: Optional[List[str]] = None):
        super().__init__(target_keys)
        self.angle_range = angle_range
        self.probability = probability

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply rotation with probability to every target data in batch.

        :param batch: batch with target keys to apply augmentation.
        :return: batch
        """
        if random.random() < self.probability:
            angle = torch.FloatTensor(1).uniform_(self.angle_range[0], self.angle_range[1]).item()
            for key in self.target_keys:
                batch[key] = Rotate.rotate_frames(batch[key], angle)

        return batch


class RandomColorJitterWithProb(BaseAug):
    def __init__(
            self,
            probability: float = 0.5,
            brightness_range: Tuple[float, float] = (1, 1),
            contrast_range: Tuple[float, float] = (1, 1),
            saturation_range: Tuple[float, float] = (1, 1),
            hue_range: Tuple[float, float] = (0, 0),
            target_keys: Optional[List[str]] = None
    ):
        """
        Color augmentation.

        :param probability: prob of applying aug.
        :param brightness_range: Tuple with min and max change, (0, 1)
        :param contrast_range: Tuple with min and max change, (0, 1)
        :param saturation_range: Tuple with min and max change, (0, 1)
        :param hue_range: Tuple with min and max change, (0, 0.5)
        :param target_keys:
        """
        super().__init__(target_keys)
        self.probability = probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

        self.color_jitter_transform = transforms.ColorJitter(
            brightness=self.brightness_range,
            contrast=self.contrast_range,
            saturation=self.saturation_range,
            hue=self.hue_range
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply apply different color augmentation to every target sample in batch.

        :param batch: batch with target keys to apply augmentation.
        :return: batch
        """
        for i in range(batch[self.target_keys[0]].shape[0]):
            for key in self.target_keys:
                if random.random() < self.probability:
                    batch[key][i] = self.color_jitter_transform(batch[key][i])

        return batch
