import os
import argparse
from utils.video_utils import extract_keyframes


def create_keyframes(dataset_path: str, n_frames: int):
    d_dirname, d_name = os.path.split(dataset_path)
    keyframes_path = os.path.join(d_dirname, f'{d_name}_{n_frames}_keyframes')
    part_dirs_list = [part_dir for part_dir in sorted(os.listdir(dataset_path))]
    for part_dir in part_dirs_list:
        videos_list = [video for video in sorted(os.listdir(os.path.join(dataset_path, part_dir)))]
        for video_name in videos_list:
            video_path = os.path.join(dataset_path, part_dir, video_name)
            dst_dir = os.path.join(keyframes_path, part_dir, video_name)
            os.makedirs(dst_dir, exist_ok=True)
            extract_keyframes(video_path, dst_dir, n_frames)
            print(f'{video_path} video is ready.')


def main():
    parser = argparse.ArgumentParser(description="Create keyframes from videos in a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("--n_frames", type=int, help="Number of keyframes to extract per video.")

    args = parser.parse_args()
    dataset_path = args.dataset_path
    n_frames = args.n_frames

    create_keyframes(dataset_path, n_frames)


if __name__ == "__main__":
    main()
