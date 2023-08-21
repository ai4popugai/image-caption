import argparse
import os
from typing import Tuple

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode

from datasets import FRAMES_KEY, LABELS_KEY
from experiments.utils import setup_run_instance
from train.train import Trainer


class InferenceDataset(Dataset):
    def __init__(self, src_dir: str, resolution: Tuple[int, int]):
        self.frame_transforms = Compose([
            ToTensor(),
            Resize(resolution, InterpolationMode.BILINEAR, antialias=False)
        ])
        dir_list = [os.path.join(src_dir, desc_dir) for desc_dir in sorted(os.listdir(src_dir))]
        self.frames_list = [os.path.join(desc_dir, os.listdir(desc_dir)[0]) for desc_dir in dir_list]

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx: int):
        frame = cv2.imread(self.frames_list[idx], cv2.IMREAD_COLOR)
        frame = self.frame_transforms(frame)
        return {FRAMES_KEY: frame, LABELS_KEY: torch.tensor(-1)}


def generate_descriptions(experiment: str, run: str, phase: str, snapshot_name: str, src_dir: str,):
    """
    Script to create description for each frame.

    :param experiment: name of the experiment (e.g., "EfficientNet_b0")
    :param run: name of the run (e.g., "run_16")
    :param phase: name of the phase (e.g., "phase_1")
    :param snapshot_name: name of the snapshot from which we take the model
    :param src_dir: directory with selected descriptions' dataset.
    :return: None
    """
    device = Trainer.get_device()

    # Setup run instance
    run_instance = setup_run_instance(experiment, run, phase)

    # Setup pretrained model
    model = run_instance.setup_pretrained_model(snapshot_name)
    model.to(device)
    model.eval()

    dataset = InferenceDataset(src_dir, run_instance.resolution)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    for batch in dataloader:
        batch = Trainer.normalize(batch, run_instance.normalizer)
        batch = Trainer.batch_to_device(batch, device)
        result = model(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new dataset using EfficientNet without the last layer.")
    parser.add_argument("--experiment", type=str, help="Name of the experiment (e.g., 'EfficientNet_b0')")
    parser.add_argument("--run", type=str, help="Name of the run (e.g., 'run_16')")
    parser.add_argument("--phase", type=str, help="Name of the phase (e.g., 'phase_1')")
    parser.add_argument("--snapshot_name", type=str, help="Name of the snapshot from which to take the model")
    parser.add_argument("--src_dir", type=str, help="Directory with selected descriptions' dataset")

    args = parser.parse_args()

    generate_descriptions(args.experiment, args.run, args.phase, args.snapshot_name, args.src_dir,)

