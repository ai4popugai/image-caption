import os

import torch
from torch.utils.data import DataLoader

from datasets.classification.gpr import GPRDataset
from experiments.EfficientNet_b0.run_16.phase_1 import Phase1
from train.train import Trainer


def create_dataset(snapshot_name: str, batch_size: int = 64, num_workers: int = 8):
    """
    Function created new dataset from existing one.
    All images from GPR dataset pass through Efficient without last layer and stores in new dataset.
    :param batch_size:
    :param num_workers: number of workers for dataloader
    :param snapshot_name: name of snapshot from which we take model
    :return:
    """
    # init run and model
    run = Phase1()
    model = run.setup_model()

    # load snapshot
    snapshot_path = os.path.join(run.snapshot_dir, snapshot_name)
    model.load_state_dict(torch.load(snapshot_path), strict=True)

    # datasets and loaders
    dataset = GPRDataset(resolution=run.resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    iterator = iter(dataloader)
    iters = len(dataloader)

    # setup device
    device = Trainer.get_device()

    # inference loop
    model.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(iters):
            batch = next(iterator)
            batch = Trainer.batch_to_device(batch, device)
            batch = Trainer.aug_loop(run.val_augs, batch)
            batch = Trainer.normalize(batch, run.normalizer)







