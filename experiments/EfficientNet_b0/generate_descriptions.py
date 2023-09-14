import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode

from datasets import FRAMES_KEY, LABELS_KEY, LOGITS_KEY
from datasets.classification.gpr import GPRDataset
from db import SQLiteDb
from experiments.utils import setup_run_instance
from train import Trainer


class InferenceDataset(Dataset):
    def __init__(self, frames_dir: str, resolution: Tuple[int, int]):
        """
        Dataset class for inference

        :param frames_dir: directory with frames
        :param resolution: resolution
        """
        self.frame_transforms = Compose([
            ToTensor(),
            Resize(resolution, InterpolationMode.BILINEAR, antialias=False)
        ])

        image_extensions = (".jpg", ".jpeg", ".png")
        self.frames_list = [os.path.join(frames_dir, file) for file in sorted(os.listdir(frames_dir)) if
                            file.endswith(image_extensions)]

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx: int):
        frame = cv2.imread(self.frames_list[idx], cv2.IMREAD_COLOR)
        frame = self.frame_transforms(frame)
        return {FRAMES_KEY: frame, LABELS_KEY: torch.tensor(-1)}


def generate_descriptions(experiment: str, run: str, phase: str, snapshot_name: str, frames_dir: str,
                          database: SQLiteDb = None,):
    """
    Script to create description for each frame.

    :param experiment: name of the experiment (e.g., "EfficientNet_b0")
    :param run: name of the run (e.g., "run_16")
    :param phase: name of the phase (e.g., "phase_1")
    :param snapshot_name: name of the snapshot from which we take the model
    :param frames_dir: directory with frames.
    :param database: database for which to write descriptions.
    :return: None
    """
    device = Trainer.get_device()

    # Setup run instance
    run_instance = setup_run_instance(experiment, run, phase)

    # Setup pretrained model
    model = run_instance.setup_pretrained_model(snapshot_name, device)
    model.to(device)
    model.eval()

    descriptions = GPRDataset().read_descriptions()
    descriptions = np.array(list(descriptions.values()))

    dataset = InferenceDataset(frames_dir, run_instance.resolution)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    class_labels = None
    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = Trainer.normalize(batch, run_instance.normalizer)
            batch = Trainer.batch_to_device(batch, device)
            result = model(batch)
            class_labels_batch = torch.argmax(result[LOGITS_KEY], dim=1).cpu()
            class_labels = class_labels_batch if class_labels is None \
                else torch.cat([class_labels, class_labels_batch], dim=0)
    class_descriptions = descriptions[class_labels.numpy()]

    if database is not None:
        video_id = os.path.basename(frames_dir)
        for (frame_path, description) in zip(dataset.frames_list, class_descriptions):
            keyframe_id = os.path.basename(frame_path)
            database.add_concept_to_row(video_id, keyframe_id, description)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new dataset using EfficientNet without the last layer.")
    parser.add_argument("--experiment", type=str, help="Name of the experiment (e.g., 'EfficientNet_b0')")
    parser.add_argument("--run", type=str, help="Name of the run (e.g., 'run_16')")
    parser.add_argument("--phase", type=str, help="Name of the phase (e.g., 'phase_1')")
    parser.add_argument("--snapshot_name", type=str, help="Name of the snapshot from which to take the model")
    parser.add_argument("--frames_dir", type=str, help="Directory with selected descriptions' dataset")

    args = parser.parse_args()

    generate_descriptions(args.experiment, args.run, args.phase, args.snapshot_name, args.frames_dir, )
