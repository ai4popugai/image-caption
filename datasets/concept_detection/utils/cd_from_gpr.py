import os
import shutil

import torch
from torch.utils.data import DataLoader

from datasets.classification.gpr import GPRDataset
from experiments.EfficientNet_b0.run_16.phase_1 import Phase1
from models.classification.feature_extractor import FeatureExtractor
from train.train import Trainer


def create_dataset(snapshot_name: str, dst_dir: str):
    """
    Function created new dataset from existing one.
    All images from GPR dataset pass through Efficient without last layer and stores in new dataset.
    :param dst_dir: directory where new dataset will be stored
    :param snapshot_name: name of snapshot from which we take model
    :return:
    """
    emd_dir = os.path.join(dst_dir, 'emd')
    os.makedirs(emd_dir, exist_ok=True)

    # init run and model
    run = Phase1()
    model = FeatureExtractor(run.setup_model())

    # load snapshot
    snapshot_path = os.path.join(run.snapshot_dir, snapshot_name)
    model.load_state_dict(torch.load(snapshot_path), strict=True)

    # datasets and loaders
    dataset = GPRDataset(resolution=run.resolution)

    # copy description
    shutil.copyfile(dataset.description_path, dst_dir)

    # setup device
    device = Trainer.get_device()

    # inference loop
    model.to(device)
    model.eval()
    with torch.no_grad():
        for ff in dataset.frames_list:
            # get id to save
            id = os.path.splitext(os.path.basename(ff))[0]

            # get batch
            ff_idx = dataset.frames_list.index(ff)
            batch = dataset[ff_idx]
            batch['frames'] = batch['frames'].unsqueeze(0)

            # prepare batch
            batch = Trainer.batch_to_device(batch, device)
            batch = Trainer.aug_loop(run.val_augs, batch)
            batch = Trainer.normalize(batch, run.normalizer)

            # inference
            result = model(batch)
