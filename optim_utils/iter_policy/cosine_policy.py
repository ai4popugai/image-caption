import math

from .base_policy import BaseIterationPolicy


class CosineAnnealingIterationPolicy(BaseIterationPolicy):
    def __init__(self, start_lr: float, start_iter: int, end_lr: float, period: int):
        self.start_lr = start_lr
        self.start_iter = start_iter
        self.end_lr = end_lr
        self.period = period

    def step(self, global_step: int) -> float:
        arg = ((global_step - self.start_iter) % self.period) / self.period  # cycles from 0 to 1
        cos = math.cos(arg * math.pi / 2)  # cycles from 1 to 0
        val = (self.start_lr - self.end_lr) * cos + self.end_lr  # cycles form start_lr to end_lr
        return val
