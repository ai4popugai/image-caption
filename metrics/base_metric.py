from abc import ABC, abstractmethod
from typing import Dict

import torch


class BaseMetric(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.unit = ''

    @abstractmethod
    def update(self, result: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def compute(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def to(self, device: torch.device) -> None:
        pass
