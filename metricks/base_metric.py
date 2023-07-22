from torch import nn


class BaseMetric(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
