import os
import shutil
import argparse

import torch
from torch.utils.data import DataLoader

from datasets.classification.gpr import GPRDataset
from models.classification.feature_extractor import FeatureExtractor
from train.train import Trainer


def create_dataset(experiment: str, run: str, phase: str, snapshot_name: str, dst_dir: str, ):
    """
    Function creates a new dataset from an existing one.
    All images from GPR dataset pass through EfficientNet without the last layer and are stored in the new dataset.
    :param dst_dir: directory where the new dataset will be stored
    :param snapshot_name: name of the snapshot from which we take the model
    :param experiment: name of the experiment (e.g., "EfficientNet_b0")
    :param run: name of the run (e.g., "run_16")
    :param phase: name of the phase (e.g., "phase_1")
    :return:
    """
    # setup device
    device = Trainer.get_device()

    emd_dir = os.path.join(dst_dir, 'emd')
    os.makedirs(emd_dir, exist_ok=True)

    # Convert experiment, run, and phase to module paths
    run_module = __import__(f'experiments.{experiment}.{run}', fromlist=[phase])
    phase_module = getattr(run_module, phase)
    run_instance = phase_module.Phase1()

    model = run_instance.setup_model()

    # load snapshot
    snapshot_path = os.path.join(run_instance.snapshot_dir, snapshot_name)
    checkpoint = torch.load(snapshot_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    # create feature extractor
    model = FeatureExtractor(model)

    # datasets and loaders
    dataset = GPRDataset(resolution=run_instance.resolution)

    # copy description
    shutil.copyfile(dataset.description_path, dst_dir)

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
            batch = Trainer.aug_loop(run_instance.val_augs, batch)
            batch = Trainer.normalize(batch, run_instance.normalizer)

            # inference
            result = model(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new dataset using EfficientNet without the last layer.")
    parser.add_argument("--experiment", type=str, help="Name of the experiment (e.g., 'EfficientNet_b0')")
    parser.add_argument("--run", type=str, help="Name of the run (e.g., 'run_16')")
    parser.add_argument("--phase", type=str, help="Name of the phase (e.g., 'phase_1')")
    parser.add_argument("--snapshot_name", type=str, help="Name of the snapshot from which to take the model")
    parser.add_argument("--dst_dir", type=str, help="Directory where the new dataset will be stored")
    args = parser.parse_args()

    create_dataset(args.experiment, args.run, args.phase, args.snapshot_name, args.dst_dir, )
