from abc import ABC, abstractmethod

from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None

    def get_model(self) -> nn.Module:
        return self.model
