import argparse
import os
import random
import re

from db import SQLiteDb
from experiments.EfficientNet_b0.generate_captions import generate_captions

# setup concept detection model constants
from keyframes_extraction import KEYFRAMES_DIR_KEY
from keyframes_extraction.extract_keyframes import extract_keyframes
from nn_models.object_detection.yolov7 import yolo_inference

EXPERIMENT = 'EfficientNet_b0'
RUN = 'run_32'
PHASE = 'phase_2'
SNAPSHOT_NAME = 'snapshot_3600.pth'


def preprocess_videos(videos_dir: str, n_frames: int, database: SQLiteDb = None, ):
    random.seed(0)

    videos_list = [video for video in sorted(os.listdir(videos_dir)) if
                   re.search('\.(mp4|avi|mov|mkv|webm)$', video, re.IGNORECASE)]

    # process videos one by one
    for video_name in videos_list:
        # setup paths
        video_path = os.path.join(videos_dir, video_name)
        keyframes_dir = os.path.join(videos_dir, KEYFRAMES_DIR_KEY, video_name)
        os.makedirs(keyframes_dir, exist_ok=True)

        # extract keyframes
        extract_keyframes(video_path, keyframes_dir, n_frames, database=database)

        # generate description
        generate_captions(EXPERIMENT, RUN, PHASE, SNAPSHOT_NAME, keyframes_dir, database=database)

        # detect objects
        yolo_inference(keyframes_dir, database=database)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract keyframes from videos')
    parser.add_argument('--videos_dir', type=str, help='Path to directory containing videos')
    parser.add_argument('--n_frames', type=int, default=20, help='Number of keyframes to extract per video')
    args = parser.parse_args()

    database = SQLiteDb(os.path.join(os.environ['SQLITE_DB_DIR'], os.path.basename(args.videos_dir)))
    database.create_db()
    preprocess_videos(args.videos_dir, args.n_frames, database)
