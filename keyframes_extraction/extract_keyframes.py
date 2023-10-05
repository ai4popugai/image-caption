import os
import argparse
import random
from datetime import timedelta
from typing import List

import cv2
import torch
import torchvision
from scenedetect import detect, ContentDetector
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor

from db import SQLiteDb
from keyframes_extraction import KEYFRAMES_DIR_KEY
from train import Trainer
from utils.video_utils.video_reader import VideoReader

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


def time_formatting(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    ms = int(td.microseconds / 1000)
    return str(td).split(".")[0].zfill(8) + f".{ms:03d}"


def extract_keyframes(video_path: str, keyframes_dir: str, n_frames: int, database: SQLiteDb = None):
    reader = VideoReader(video_path)
    scene_list = detect(video_path, ContentDetector())
    if len(scene_list) == 0:
        # fallback for case if scene detect algorythm is failed
        step = reader.frames_count // n_frames
        for i in range(n_frames):
            keyframe_index = i * step
            keyframe_id = f'frame_{"%05d" % keyframe_index}.png'
            keyframe = reader[keyframe_index]
            cv2.imwrite(os.path.join(keyframes_dir, keyframe_id), keyframe)
            if database is not None:
                database.add_new_row(video_id=os.path.basename(video_path), keyframe_id=keyframe_id,
                                     timestamp=time_formatting(keyframe_index/reader.fps))
        return

    n_per_scene = n_frames // len(scene_list) + 1

    # form keyframes list
    keyframes_id_list = []
    for scene in scene_list:
        scene_start_idx = scene[0].frame_num
        scene_end_idx = scene[1].frame_num - 1
        scene_len = scene_end_idx - scene_start_idx
        step = scene_len // n_per_scene
        for i in range(n_per_scene):
            keyframe_id = scene_start_idx + i * step
            keyframes_id_list.append(keyframe_id)

    random_n_frames_indexes = sorted(random.sample(keyframes_id_list, n_frames))

    # store selected keyframes
    for keyframe_index in random_n_frames_indexes:
        keyframe_id = f'frame_{"%05d" % keyframe_index}.png'
        keyframe = reader[keyframe_index]
        cv2.imwrite(os.path.join(keyframes_dir, keyframe_id), keyframe)
        if database is not None:
            database.add_new_row(video_id=os.path.basename(video_path), keyframe_id=keyframe_id,
                                 timestamp=time_formatting(keyframe_index/reader.fps))


def extract_keyframes_for_dataset(dataset_path: str, n_frames: int,):
    random.seed(0)

    videos_list = [video for video in sorted(os.listdir(dataset_path)) if video.endswith('.mp4')]

    for video_name in videos_list:
        # setup paths
        video_path = os.path.join(dataset_path, video_name)
        keyframes_dir = os.path.join(dataset_path, KEYFRAMES_DIR_KEY, video_name)
        os.makedirs(keyframes_dir, exist_ok=True)

        # extract keyframes
        extract_keyframes(video_path, keyframes_dir, n_frames)


def main():
    parser = argparse.ArgumentParser(description="Create keyframes from videos in a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("--n_frames", type=int, help="Desired number of keyframes.")

    args = parser.parse_args()

    extract_keyframes_for_dataset(args.dataset_path, args.n_frames)


if __name__ == "__main__":
    main()
