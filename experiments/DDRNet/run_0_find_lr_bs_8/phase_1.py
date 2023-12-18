import os

from experiments.DDRNet.run_base import RunBase
from optim_utils.iter_policy.linear_policy import LinearIterationPolicy


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.batch_size = 8
        self.num_workers = 6

        self.train_iters = 1000
        self.batch_dump_iters = 1000
        self.snapshot_iters = 1000
        self.show_iters = 10

        self.optimizer_kwargs = {'lr': 0., 'weight_decay': 3e-5}
        self.lr_policy = LinearIterationPolicy(start_iter=0, start_val=0, end_iter=20000, end_val=3e-3)


if __name__ == '__main__':
    Phase().train(start_snapshot=None)