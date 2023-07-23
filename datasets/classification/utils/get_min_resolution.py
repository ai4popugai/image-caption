import os

from PIL import Image


def main():
    if os.getenv("GPR_DATASET") is None:
        raise RuntimeError('Dataset path must be set up.')
    root = os.environ['GPR_DATASET']

    frames_list = [os.path.join(root, file_name) for file_name in sorted(os.listdir(root))]
    min_height, min_width = 1024, 2048
    for frame_path in frames_list:
        frame = Image.open(frame_path)
        min_height = min(min_height, frame.height)
        min_width = min(min_width, frame.width)
    print(f'min_height: {min_height}, min_width: {min_width}')


if __name__ == '__main__':
    main()
