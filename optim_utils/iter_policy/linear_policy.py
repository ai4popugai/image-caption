from .base_policy import BaseIterationPolicy


class LinearIterationPolicy(BaseIterationPolicy):
    def __init__(self, start_iter: int, start_val: float, end_iter: int, end_val: float):
        self.start_iter = start_iter
        self.start_val = start_val
        self.end_iter = end_iter
        self.end_val = end_val

    def evaluate(self, global_step: int) -> float:
        if global_step < self.start_iter:
            return self.start_val
        elif global_step >= self.end_iter:
            return self.end_val

        arg = (global_step - self.start_iter) / (self.end_iter - self.start_iter)
        val = self.start_val + (self.end_val - self.start_val) * arg
        return val
