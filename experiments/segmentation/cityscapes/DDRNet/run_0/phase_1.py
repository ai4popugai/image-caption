import os

from experiments.segmentation.cityscapes.DDRNet.run_base import RunBase


class Phase(RunBase):
    def __init__(self):
        super().__init__(os.path.abspath(__file__))

        self.optimizer_kwargs = {'lr': 5e-4, 'weight_decay': 5e-4}


if __name__ == '__main__':
    Phase().train(start_snapshot=None)
