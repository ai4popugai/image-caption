import argparse
import os
import random

from datasets.video.v3c1.utils.extract_keyframes import extract_keyframes
from db import SQLiteDb
from web import KEYFRAMES_DIR_KEY


def main(videos_dir: str, n_frames: int):
    random.seed(0)

    videos_list = [video for video in sorted(os.listdir(videos_dir)) if video.endswith('.mp4')]

    database = SQLiteDb(os.path.basename(videos_dir))
    # process videos one by one
    for video_name in videos_list:
        # setup paths
        video_path = os.path.join(videos_dir, video_name)
        keyframes_path = os.path.join(videos_dir, KEYFRAMES_DIR_KEY, video_name)
        os.makedirs(keyframes_path, exist_ok=True)

        # extract keyframes
        extract_keyframes(video_path, keyframes_path, n_frames, database=database)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract keyframes from videos')
    parser.add_argument('--videos_dir', type=str, help='Path to directory containing videos')
    parser.add_argument('--n_frames', type=int, default=10, help='Number of keyframes to extract per video')
    args = parser.parse_args()

    main(args.videos_dir, args.n_frames)
