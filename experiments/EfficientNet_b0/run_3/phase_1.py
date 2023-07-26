import os

from experiments.EfficientNet_b0.run_base import RunBase
from optim_utils.iter_policy.linear_policy import LinearIterationPolicy


class Phase1(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'weight_decay': 3e-5}
        self.lr_policy = LinearIterationPolicy(0, 0, 3000, 6e-4)

        self.max_iteration = 18000


if __name__ == '__main__':
    Phase1().train(start_snapshot_name=None)
