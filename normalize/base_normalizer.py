from abc import ABC, abstractmethod

from torchvision import transforms


class BaseNormalizer(ABC):
    def __init__(self, normalizer: transforms.Normalize):
        super().__init__()
        self.normalizer = normalizer

    @abstractmethod
    def __call__(self, batch):
        raise NotImplementedError('Method should be implemented in child class')
