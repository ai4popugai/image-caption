from .base_policy import BaseIterationPolicy


class LinearIterationPolicy(BaseIterationPolicy):
    def __init__(self, start_iter: int, start_lr: float, end_iter: int, end_lr: float):
        self.start_iter = start_iter
        self.start_lr = start_lr
        self.end_iter = end_iter
        self.end_lr = end_lr

    def step(self, global_step: int) -> float:
        if global_step < self.start_iter:
            return self.start_lr
        elif global_step >= self.end_iter:
            return self.end_lr

        arg = (global_step - self.start_iter) / (self.end_iter - self.start_iter)
        val = self.start_lr + (self.end_lr - self.start_lr) * arg
        return val
