import os
import argparse
import random
from typing import List

import cv2
import torch
import torchvision
from scenedetect import detect, ContentDetector, FrameTimecode
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor

from datasets.video.v3c1.utils import KEYFRAMES_DIR_KEY
from db import SQLiteDb
from train.train import Trainer
from utils.video_utils.video_reader import VideoReader

FPS = 25

# setup pretrained embeddings extractor
DEVICE = Trainer.get_device()
WEIGHTS = torchvision.models.ResNet50_Weights.DEFAULT
MODEL = torchvision.models.resnet50(weights=WEIGHTS)
MODEL = nn.Sequential(*list(MODEL.children())[:-1])
MODEL = MODEL.to(DEVICE)
MODEL.eval()
PREPROCESS = WEIGHTS.transforms()


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


def extract_keyframes(video_path: str, dst_dir: str, n_frames: int, database: SQLiteDb = None):
    reader = VideoReader(video_path, fps=FPS)
    scene_list = detect(video_path, ContentDetector())
    if len(scene_list) == 0:
        # create by my own
        step = reader.frames_count // n_frames
        for i in range(n_frames):
            keyframe_index = i * step
            keyframe_id = f'frame_{"%05d" % keyframe_index}.png'
            keyframe = reader[keyframe_index]
            cv2.imwrite(os.path.join(dst_dir, keyframe_id), keyframe)
            if database is not None:
                database.add_new_key(video_id=os.path.basename(video_path), keyframe_id=keyframe_id)
        return

    # cut the first and the last scenes if it is possible
    if len(scene_list) >= n_frames + 6:
        scene_list = scene_list[3:-3]
    elif len(scene_list) >= n_frames + 4:
        scene_list = scene_list[2:-2]
    elif len(scene_list) >= n_frames + 2:
        scene_list = scene_list[1:-1]

    n_per_scene = n_frames // len(scene_list) + 1

    # form keyframes list
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
        batch = PREPROCESS(batch)
        batch = batch.to(DEVICE)
        out = MODEL(batch).reshape(batch.shape[0], -1).cpu().detach()
        features = out if features is None else torch.cat([features, out], dim=0)

    # ensure diversity
    kmeans = KMeans(n_clusters=n_frames, random_state=0)
    cluster_assignments = kmeans.fit_predict(features)

    # store selected keyframes
    saved_clusters = []
    for i, cluster_id in enumerate(cluster_assignments):
        if cluster_id not in saved_clusters:
            saved_clusters.append(cluster_id)
            keyframe_index = keyframes_id_list[i]
            keyframe_id = f'frame_{"%05d" % keyframe_index}.png'
            keyframe = reader[keyframe_index]
            cv2.imwrite(os.path.join(dst_dir, keyframe_id), keyframe)
            if database is not None:
                database.add_new_key(video_id=os.path.basename(video_path), keyframe_id=keyframe_id)
            if saved_clusters == n_frames:
                break


def extract_keyframes_for_dataset(dataset_path: str, n_frames: int,):
    random.seed(0)

    videos_list = [video for video in sorted(os.listdir(dataset_path)) if video.endswith('.mp4')]

    for video_name in videos_list:
        # setup paths
        video_path = os.path.join(dataset_path, video_name)
        keyframes_path = os.path.join(dataset_path, KEYFRAMES_DIR_KEY, video_name)
        os.makedirs(keyframes_path, exist_ok=True)

        # extract keyframes
        extract_keyframes(video_path, keyframes_path, n_frames)


def main():
    parser = argparse.ArgumentParser(description="Create keyframes from videos in a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("--n_frames", type=int, help="Desired number of keyframes.")

    args = parser.parse_args()

    extract_keyframes_for_dataset(args.dataset_path, args.n_frames)


if __name__ == "__main__":
    main()
