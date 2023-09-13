import os
import random

from datasets.video.v3c1.utils.extract_keyframes import extract_keyframes
from web import KEYFRAMES_DIR_KEY


def main(videos_dir: str, n_frames: int,):
    random.seed(0)

    videos_list = [video for video in sorted(os.listdir(videos_dir)) if video.endswith('.mp4')]
    # process videos one by one
    for video_name in videos_list:
        # setup paths
        video_path = os.path.join(videos_dir, video_name)
        keyframes_path = os.path.join(videos_dir, KEYFRAMES_DIR_KEY, video_name)
        os.makedirs(keyframes_path, exist_ok=True)

        # extract keyframes
        extract_keyframes(video_path, keyframes_path, n_frames)
