import os
import argparse
import random
from typing import List

import lpips
import torch
from scenedetect import detect, ContentDetector, FrameTimecode
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor

from utils.video_utils.video_reader import VideoReader

FPS = 25


class KeyFramesDataset(Dataset):
    def __init__(self, keyframes_id_list: List[int], reader: VideoReader):
        super().__init__()
        self.keyframes_id_list = keyframes_id_list
        self.reader = reader
        self.frame_transforms = Compose([
            ToTensor(),
            Resize((128, 128), InterpolationMode.BILINEAR, antialias=False)
        ])

    def __len__(self):
        return len(self.keyframes_id_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.reader[self.keyframes_id_list[idx]]
        frame = self.frame_transforms(frame)
        return frame


def extract_keyframes(dataset_path: str, n_frames: int, batch_size: int = 8):
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

            # get pretrained to perceptual loss new
            loss_fn = lpips.LPIPS(net='alex')
            net = loss_fn.net
            scaler = loss_fn.scaling_layer
            del loss_fn

            dataset = KeyFramesDataset(keyframes_id_list, reader)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            features = None
            for batch in enumerate(dataloader):
                out = net(scaler(batch)).reshape(batch_size, -1)
                features = out if features is None else torch.cat([features, out], dim=0)

            print(f'{video_path} video is ready.')


def main():
    parser = argparse.ArgumentParser(description="Create keyframes from videos in a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("--n_frames", type=int, help="Desired number of keyframes.")
    parser.add_argument("--batch_size", type=int, help="Batch size to compute image features.")

    args = parser.parse_args()

    extract_keyframes(args.dataset_path, args.n_frames, args.batch_size)


if __name__ == "__main__":
    main()
