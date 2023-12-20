from .base_policy import BaseIterationPolicy


class LinearIterationPolicy(BaseIterationPolicy):
    def __init__(self, start_iter: int, start_lr: float, end_lr: int, end_val: float):
        self.start_iter = start_iter
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.end_val = end_val

    def step(self, global_step: int) -> float:
        if global_step < self.start_iter:
            return self.start_lr
        elif global_step >= self.end_lr:
            return self.end_val

        arg = (global_step - self.start_iter) / (self.end_lr - self.start_iter)
        val = self.start_lr + (self.end_val - self.start_lr) * arg
        return val
