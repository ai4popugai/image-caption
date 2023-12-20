import torch

from datasets.segmantation.cityscapes import logits_to_colors, activation_map_to_colors


class BaseToImageTransforms:
    def __init__(self):
        """
        Base class to all sample to image transforms.
        """
        self.permute_dims = (0, 2, 3, 1)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class CityscapesLogitsToImage(BaseToImageTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, logits: torch.Tensor):
        """
        Method to convert logits of segmentation to image

        :param logits: logits with shape [batch_size, N_CLASSES, h, w].
        :return:
        """
        return logits_to_colors(logits).permute(self.permute_dims).cpu().numpy()


class CityscapesGroundTruthToImage(BaseToImageTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, logits: torch.Tensor):
        """
        Method to convert segmentation ground truth to image

        :param logits: ground truth with shape [batch_size, h, w].
        :return:
        """
        return activation_map_to_colors(logits).permute(self.permute_dims).cpu().numpy()


class CityscapesFramesToImage(BaseToImageTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, frames: torch.Tensor):
        """
        Method to convert frames represented as torch.Tensor to image

        :param frames: frames with shape [batch_size, CHANNELS, h, w].
        :return:
        """
        return (frames * 255).to(torch.uint8).permute(self.permute_dims).cpu().numpy()
