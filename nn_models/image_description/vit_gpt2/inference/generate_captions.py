import argparse
import os

import torch
from torch.utils.data import Dataset, DataLoader

from db import SQLiteDb
from nn_models.image_description.vit_gpt2.inference.inference import predict_step


class DescriptionsDataset(Dataset):
    def __init__(self, frames_dir: str):
        self.frames_dir = frames_dir
        image_extensions = (".jpg", ".jpeg", ".png")
        self.frames_list = [os.path.join(frames_dir, file) for file in sorted(os.listdir(frames_dir)) if
                            file.endswith(image_extensions)]

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx: int) -> str:
        return self.frames_list[idx]


def generate_descriptions(frames_dir: str,
                          database: SQLiteDb = None,):
    """
    Functon to generate descriptions for images in directory.

    :param frames_dir: directory with images.
    :param database: database to write descriptions.
    :return: None
    """
    dataset = DescriptionsDataset(frames_dir)
    dataloader = DataLoader(dataset, batch_size=16)

    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            captions = predict_step(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image descriptions.")
    parser.add_argument("--snapshot_name", type=str, help="Name of the snapshot from which to take the model")
    parser.add_argument("--frames_dir", type=str, help="Directory with selected descriptions' dataset")

    args = parser.parse_args()

    generate_descriptions(args.frames_dir,)
