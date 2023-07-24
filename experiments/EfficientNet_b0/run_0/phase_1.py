from experiments.EfficientNet_b0.run_base import RunBase


class Phase1(RunBase):
    def __init__(self):
        super().__init__(__file__)

        self.optimizer_kwargs = {'lr': 3e-5, 'weight_decay': 3e-7}
        self.lr_policy = LinearIterationPolicy(0, 0, 5000, 3.0e-4 / 3.2)  # batch size changed from 16 to 5

