import os

from experiments.DDRNet.run_base import RunBase
from optim_utils.iter_policy.cosine_policy import CosineAnnealingIterationPolicy


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'lr': 0., 'weight_decay': 3e-5}
        self.lr_policy = CosineAnnealingIterationPolicy(start_iter=16000, start_lr=2.4e-3, end_lr=0, period=5000)


if __name__ == '__main__':
    start_snapshot = 'DDRNet/run_0_find_lr_bs_8/DDRNet23Slim_16000.pth'
    Phase().train(start_snapshot=start_snapshot)
