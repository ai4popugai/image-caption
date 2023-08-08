import os
import argparse
import random

import cv2
from scenedetect import detect, ContentDetector, FrameTimecode

from utils.video_utils.video_reader import VideoReader

FPS = 25


def extract_keyframes(dataset_path: str):
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
            reader = VideoReader(video_path, fps=FPS)

            for scene in scene_list:
                scene_start_idx = scene[0].frame_num
                scene_end_idx = scene[1].frame_num
                scene_len = scene_end_idx - scene_start_idx
                scene_middle_idx = (scene_start_idx + scene_end_idx) // 2

                # get idx from 1/4 to 3/4 of scene
                keyframe_id = random.randint(scene_middle_idx - scene_len // 4,
                                             scene_middle_idx + scene_len // 4)
                keyframe = reader[keyframe_id]
                cv2.imwrite(os.path.join(dst_dir, f'frame_{"%05d" % keyframe_id}.png'), keyframe)

            print(f'{video_path} video is ready.')


def main():
    parser = argparse.ArgumentParser(description="Create keyframes from videos in a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset directory.")

    args = parser.parse_args()

    extract_keyframes(args.dataset_path)


if __name__ == "__main__":
    main()
