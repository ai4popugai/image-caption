import argparse
import datetime
import os
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from datasets.segmantation.cityscapes import CityscapesVideoDataset, CITYSCAPES_VIDEO_ROOT
from train import Trainer


def main(dst_path: str, optical_flow_model: torch.nn.Module, loader: DataLoader, device: torch.device, skip_iters: int):

    iterator = iter(loader)
    num_iters = len(loader)
    for global_step in range(num_iters):
        if global_step < skip_iters:
            continue
        batch = next(iterator)
        batch = Trainer.batch_to_device(batch, device)
        flow: Dict = optical_flow_model(batch)
        for i, flow_map_batch in enumerate(flow.values()):
            step = global_step + i
            for flow_map in flow_map_batch:
                torch.save(flow_map.to('cpu'), f'{dst_path}/{"%05d" % step}.pt')
                print(f'{step}: {flow_map.shape}{7 * " "}{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    if CITYSCAPES_VIDEO_ROOT not in os.environ:
        raise RuntimeError(
            f'Failed to init cityscapes dataset instance: {CITYSCAPES_VIDEO_ROOT} not in environment.')

    parser = argparse.ArgumentParser(description='Create optical flow dataset from cityscapes dataset')
    parser.add_argument('--step', type=int, help='Step between 2 frames  in dataset', default=1)
    parser.add_argument('--optical_flow_model', type=str, help='Model to compute optical flow', default='Raft')
    parser.add_argument('--optical_flow_model_kwargs', type=Dict[str, Any], help='Model to compute optical flow',
                        default={'small': False})
    parser.add_argument('--batch_size', type=int, help='batch size for dataloader to run run optical flow model',
                        default=1)
    parser.add_argument('--num_workers', type=int, help='num workers for dataloader to run run optical flow model',
                        default=4)
    parser.add_argument('--device', type=str, help='device to inference optical flow model')

    args = parser.parse_args()

    dst_path = os.path.join(os.environ[CITYSCAPES_VIDEO_ROOT], 'optical_flow')
    dst_path = f'{dst_path}_step-{args.step}_model-{args.optical_flow_model}'
    for key, val in args.optical_flow_model_kwargs.items():
        dst_path += f'_{key}-{val}'
    os.makedirs(dst_path, exist_ok=True)
    file_names = sorted(os.listdir(dst_path))
    if len(file_names) != 0:
        skip_iters = int(file_names[-1].split('.')[0]) // args.batch_size
    else:
        skip_iters = 0
    dataset = CityscapesVideoDataset(step=args.step)
    device = torch.device(args.device)
    optical_flow_module = __import__(f'nn_models.optical_flow', fromlist=[''])
    optical_flow_model = getattr(optical_flow_module, args.optical_flow_model)\
        (**args.optical_flow_model_kwargs).to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    main(dst_path, optical_flow_model, loader, device, skip_iters)
