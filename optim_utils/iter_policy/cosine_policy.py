import math

from optim_utils.iter_policy.base_policy import BaseIterationPolicy
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    policy = CosineAnnealingIterationPolicy(start_lr=1e-1, start_iter=0, end_lr=1e-3, period=1000)
    items = []
    for i in range(10000):
        items.append(policy.step(i))

    plt.plot(items)
    plt.show()

