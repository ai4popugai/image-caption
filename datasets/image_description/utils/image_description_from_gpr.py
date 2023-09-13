import os
import shutil
import argparse

import torch

from datasets import FEATURE_MAPS_KEYS, FRAMES_KEY, LABELS_KEY
from datasets.classification.gpr import GPRDataset
from experiments.EfficientNet_b0.efficient_net_b0 import EfficientNetFeatureMapExtractor, EfficientNet
from experiments.utils import setup_run_instance
from train import Trainer


def create_dataset(experiment: str, run: str, phase: str, snapshot_name: str, dataset_dir: str, name_prefix: str = '',
                   num_removed_layers: int = 4):
    """
    Function creates a new dataset from an existing one.
    All images from GPR dataset pass through EfficientNet without the last layer and are stored in the new dataset.
    :param num_removed_layers: number of layers to leave in the model from start.
    :param name_prefix: prefix for the name of the new dataset. If empty, the name of the snapshot is used.
    :param dataset_dir: directory where the new dataset will be stored
    :param snapshot_name: name of the snapshot from which we take the model
    :param experiment: name of the experiment (e.g., "EfficientNet_b0")
    :param run: name of the run (e.g., "run_16")
    :param phase: name of the phase (e.g., "phase_1")
    :return:
    """
    # setup device
    device = Trainer.get_device()
    # device = torch.device('cpu')

    dst_dir = os.path.join(dataset_dir, f'{name_prefix}_{run}_{phase}_{os.path.splitext(snapshot_name)[0]}'
                                        f'_num_removed_layers_{num_removed_layers}')
    emd_dir = os.path.join(dst_dir, FEATURE_MAPS_KEYS)
    os.makedirs(emd_dir, exist_ok=True)

    # Setup run instance
    run_instance = setup_run_instance(experiment, run, phase)

    # Setup pretrained model
    model = run_instance.setup_pretrained_model(snapshot_name)

    # create feature extractor
    model = EfficientNetFeatureMapExtractor(model, num_removed_layers=num_removed_layers)

    # datasets and loaders
    dataset = GPRDataset(resolution=run_instance.resolution)

    # copy description
    shutil.copy(dataset.description_path, dst_dir)

    # inference loop
    model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, ff in enumerate(dataset.frames_list):
            # get id to save
            id = os.path.splitext(os.path.basename(ff))[0]

            # get batch
            batch = dataset[idx]
            batch[FRAMES_KEY] = batch[FRAMES_KEY].unsqueeze(0)
            batch[LABELS_KEY] = batch[LABELS_KEY].unsqueeze(0)

            # prepare batch
            batch = Trainer.batch_to_device(batch, device)
            batch = Trainer.aug_loop(batch, run_instance.val_augs)
            batch = Trainer.normalize(batch, run_instance.normalizer)

            # inference
            result = model(batch)
            torch.save(result[FEATURE_MAPS_KEYS].squeeze(), os.path.join(emd_dir, f'{id}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new dataset using EfficientNet without the last layer.")
    parser.add_argument("--experiment", type=str, help="Name of the experiment (e.g., 'EfficientNet_b0')")
    parser.add_argument("--run", type=str, help="Name of the run (e.g., 'run_16')")
    parser.add_argument("--phase", type=str, help="Name of the phase (e.g., 'phase_1')")
    parser.add_argument("--snapshot_name", type=str, help="Name of the snapshot from which to take the model")
    parser.add_argument("--dataset_dir", type=str, help="Directory where the new dataset will be stored")
    parser.add_argument("--name_prefix", type=str, help='Prefix for the name of the new dataset')
    parser.add_argument("--num_removed_layers", type=str, default=4, help='Num layers to leave from start')
    args = parser.parse_args()

    create_dataset(args.experiment, args.run, args.phase, args.snapshot_name, args.dataset_dir, args.name_prefix,
                   args.num_removed_layers,)
