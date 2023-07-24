from torch.optim import Optimizer

from optim_utils.iter_policy.base_policy import BaseIterationPolicy


class LrPolicy:
    def __init__(self, optimizer: Optimizer, policy: BaseIterationPolicy):
        self.optimizer = optimizer
        self.policy = policy

    def step(self, global_step: int):
        """
        Sets current learning rate to the optimizer

        :param global_step - iteration counter.
        :return learning rate value used.
        """
        lr = self.policy.step(global_step)
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr
