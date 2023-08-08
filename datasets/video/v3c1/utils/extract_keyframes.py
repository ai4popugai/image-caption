import os
import argparse
import random
from typing import List

import cv2
from scenedetect import detect, ContentDetector, FrameTimecode
from torch import nn

from utils.video_utils.video_reader import VideoReader

FPS = 25


class KeyFramesDataset(nn.Module):
    def __init__(self, keyframes_id_list: List[int], reader: VideoReader):
        super().__init__()
        self.keyframes_id_list = keyframes_id_list
        self.reader = reader

    def __len__(self):
        return len(self.keyframes_id_list)

    def __getitem__(self, idx: int):
        return self.reader[self.keyframes_id_list[idx]]


def extract_keyframes(dataset_path: str, n_frames: int):
    d_dirname, d_name = os.path.split(dataset_path)
    keyframes_path = os.path.join(d_dirname, f'{d_name}_keyframes')
    part_dirs_list = [part_dir for part_dir in sorted(os.listdir(dataset_path))]
    for part_dir in part_dirs_list:
        videos_list = [video for video in sorted(os.listdir(os.path.join(dataset_path, part_dir)))]
        for video_name in videos_list:
            video_path = os.path.join(dataset_path, part_dir, video_name)
            dst_dir = os.path.join(keyframes_path, part_dir, video_name)
            os.makedirs(dst_dir, exist_ok=True)

            scene_list = detect(video_path, ContentDetector())
            scene_list = scene_list[1:-1]

            n_per_scene = n_frames // len(scene_list) + 1
            reader = VideoReader(video_path, fps=FPS)

            keyframes_id_list = []
            for scene in scene_list:
                scene_start_idx = scene[0].frame_num
                scene_end_idx = scene[1].frame_num
                for _ in range(n_per_scene):
                    keyframe_id = random.randint(scene_start_idx, scene_end_idx)
                    keyframes_id_list.append(keyframe_id)
                # keyframe = reader[keyframe_id]
                # cv2.imwrite(os.path.join(dst_dir, f'frame_{"%05d" % keyframe_id}.png'), keyframe)

            print(f'{video_path} video is ready.')


def main():
    parser = argparse.ArgumentParser(description="Create keyframes from videos in a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset directory.")

    args = parser.parse_args()

    extract_keyframes(args.dataset_path)


if __name__ == "__main__":
    main()
