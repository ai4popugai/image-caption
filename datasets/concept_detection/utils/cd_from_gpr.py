import torch
from torch.utils.data import DataLoader

from datasets.classification.gpr import NUM_CLASSES, GPRDataset
from experiments.EfficientNet_b0.efficient_net_b0 import EfficientNet
from train.train import Trainer


def create_dataset(snapshot_path):
    """
    Function created new dataset from existing one.
    All images from GPR dataset pass through Efficient without last layer and stores in new dataset.
    :param snapshot_path:
    :return:
    """
    model = EfficientNet(NUM_CLASSES)
    model.load_state_dict(torch.load(snapshot_path))
    model.eval()
    device = Trainer.get_device()
    model.to(device)

    dataset = GPRDataset(resolution=(256, 256))

    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8)
    iterator = iter(dataloader)
    iters = len(dataloader)

    for _ in range(iters):
        batch = next(iterator)
        frames = batch['frames'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            features = model.extract_features(frames)
        dataset.add_features(features, labels)





