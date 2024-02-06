import torch


def logits_to_activation_map(logits: torch.Tensor) -> torch.Tensor:
    """
    Method converts logits [batch_size, N_CLASSES, h, w] to activation map with class label in dim 1
    [batch_size, h, w].

    :param logits: segmentation logits
    :return: tensor with class labels
    """
    segmentations = torch.argmax(logits.clone().detach(), dim=1)
    return segmentations


def activation_map_to_colors(activation_map: torch.Tensor, color_map: torch.Tensor) -> torch.Tensor:
    """
    Converts activation map [batch_size, h, w] with classes in 1 dim
    to tensor [batch_size, CHANNELS, h, w] with colors in 1 dim.

    :param color_map: tensor with BRG colors for corresponded  classes of dataset
    :param activation_map: segmentation activation map
    :return: color tensor.
    """
    mapped = color_map.to(activation_map.device)[activation_map.long()].permute(0, 3, 1, 2)
    return mapped


def logits_to_colors(logits: torch.Tensor, color_map: torch.Tensor) -> torch.Tensor:
    """
    Method converts logits [batch_size, N_CLASSES, h, w]
    to tensor [batch_size, CHANNELS, h, w] with colors in 1 dim.

    :param color_map: tensor with BRG colors for corresponded  classes of dataset
    :param logits: segmentation logits
    :return:
    """
    return activation_map_to_colors(logits_to_activation_map(logits), color_map=color_map)


class BaseToImageTransforms:
    def __init__(self):
        """
        Base class to all sample to image transforms.
        """
        self.permute_dims = (0, 2, 3, 1)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LogitsToImage(BaseToImageTransforms):
    def __init__(self, color_map: torch.Tensor):
        """
        Method to convert segmentation logits to numpy array images.

        :param color_map: tensor with BRG colors for corresponded  classes of dataset
        """
        self.color_map = color_map
        super().__init__()

    def __call__(self, logits: torch.Tensor):
        """
        Method to convert logits of segmentation to image

        :param logits: logits with shape [batch_size, N_CLASSES, h, w].
        :return:
        """
        return logits_to_colors(logits, self.color_map).permute(self.permute_dims).cpu().numpy()


class GroundTruthToImage(BaseToImageTransforms):
    def __init__(self, color_map: torch.Tensor):
        """
        Method to convert segmentation ground truths to numpy array images.

        :param color_map:
        """
        self.color_map = color_map
        super().__init__()

    def __call__(self, ground_truth: torch.Tensor):
        """
        Method to convert segmentation ground truth to image

        :param ground_truth: ground truth with shape [batch_size, h, w].
        :return:
        """
        return activation_map_to_colors(ground_truth, self.color_map).permute(self.permute_dims).cpu().numpy()


class FramesFloatToImage(BaseToImageTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, frames: torch.Tensor):
        """
        Method to convert frames represented as torch.Tensor with values range [0, 1] to image

        :param frames: frames with shape [batch_size, CHANNELS, h, w].
        :return:
        """
        return (frames.clone().detach() * 255).to(torch.uint8).permute(self.permute_dims).cpu().numpy()


class FramesIntToImage(BaseToImageTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, frames: torch.Tensor):
        """
        Method to convert frames represented as torch.Tensor with values range [0, 255] to image

        :param frames: frames with shape [batch_size, CHANNELS, h, w].
        :return:
        """
        if frames.dtype == torch.float32:
            frames = torch.round(frames).to(torch.uint8)

        return frames.clone().detach().permute(self.permute_dims).cpu().numpy()

