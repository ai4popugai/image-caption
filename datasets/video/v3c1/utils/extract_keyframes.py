import os
import argparse
import random
from typing import List

import cv2
import lpips
import torch
import torchvision
from scenedetect import detect, ContentDetector, FrameTimecode
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor

from train.train import Trainer
from utils.video_utils.video_reader import VideoReader

FPS = 25


class KeyFramesDataset(Dataset):
    def __init__(self, keyframes_id_list: List[int], reader: VideoReader):
        super().__init__()
        self.keyframes_id_list = keyframes_id_list
        self.reader = reader
        self.frame_transforms = Compose([
            ToTensor(),
            Resize((256, 256), InterpolationMode.BILINEAR, antialias=False)
        ])

    def __len__(self):
        return len(self.keyframes_id_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.reader[self.keyframes_id_list[idx]]
        frame = self.frame_transforms(frame)
        return frame


def extract_keyframes(dataset_path: str, n_frames: int,):
    # setup device
    device = Trainer.get_device()
    random.seed(0)

    # get pretrained to perceptual loss new
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    preprocess = weights.transforms()

    d_dirname, d_name = os.path.split(dataset_path)
    keyframes_path = os.path.join(d_dirname, f'{d_name}_keyframes')
    part_dirs_list = [part_dir for part_dir in sorted(os.listdir(dataset_path))]
    for part_dir in part_dirs_list:
        videos_list = [video for video in sorted(os.listdir(os.path.join(dataset_path, part_dir)))]
        for video_name in videos_list:
            video_path = os.path.join(dataset_path, part_dir, video_name)
            dst_dir = os.path.join(keyframes_path, part_dir, video_name)
            os.makedirs(dst_dir, exist_ok=False)

            scene_list = detect(video_path, ContentDetector())
            if len(scene_list) >= n_frames + 6:
                scene_list = scene_list[3:-3]
            elif len(scene_list) >= n_frames + 4:
                scene_list = scene_list[2:-2]
            elif len(scene_list) >= n_frames + 2:
                scene_list = scene_list[1:-1]

            n_per_scene = n_frames // len(scene_list) + 1
            reader = VideoReader(video_path, fps=FPS)

            keyframes_id_list = []
            for scene in scene_list:
                scene_start_idx = scene[0].frame_num
                scene_end_idx = scene[1].frame_num - 1
                for _ in range(n_per_scene):
                    keyframe_id = random.randint(scene_start_idx, scene_end_idx)
                    keyframes_id_list.append(keyframe_id)

            dataset = KeyFramesDataset(keyframes_id_list, reader)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

            features = None
            for batch in dataloader:
                batch = preprocess(batch)
                batch = batch.to(device)
                out = model(batch).reshape(batch.shape[0], -1).cpu().detach()
                features = out if features is None else torch.cat([features, out], dim=0)

            kmeans = KMeans(n_clusters=n_frames, random_state=0)
            cluster_assignments = kmeans.fit_predict(features)

            saved_clusters = []
            for i, cluster_id in enumerate(cluster_assignments):
                if cluster_id not in saved_clusters:
                    saved_clusters.append(cluster_id)
                    keyframe_id = keyframes_id_list[i]
                    keyframe = reader[keyframe_id]
                    cv2.imwrite(os.path.join(dst_dir, f'frame_{"%05d" % keyframe_id}.png'), keyframe)
                    if saved_clusters == n_frames:
                        break

            print(f'{video_path} video is ready.')


def main():
    parser = argparse.ArgumentParser(description="Create keyframes from videos in a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("--n_frames", type=int, help="Desired number of keyframes.")

    args = parser.parse_args()

    extract_keyframes(args.dataset_path, args.n_frames)


if __name__ == "__main__":
    main()
