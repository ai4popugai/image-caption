import argparse
import glob
import json
import os

from yolov7_package import Yolov7Detector
import cv2


def inference(frames_dir: str, threshold: float = 0.5):
    det = Yolov7Detector(traced=False)

    image_extensions = (".jpg", ".jpeg", ".png")
    frames_list = [os.path.join(frames_dir, file) for file in sorted(os.listdir(frames_dir)) if
                   file.endswith(image_extensions)]

    for frame_path in frames_list:
        img = cv2.imread(frame_path)
        classes, boxes, scores = det.detect(img)
        objects = []
        for class_id, box, score in zip(classes[0], boxes[0], scores[0]):
            if score >= threshold:
                objects.append({det.names[class_id]: box})
                img = det.draw_on_image(img, [box], [score], [class_id])
        store_path = os.path.join(frames_dir, f'{os.path.basename(frame_path)}_objects.json')
        if os.path.isfile(store_path):
            os.remove(store_path)
        with open(store_path, 'w') as f:
            json.dump(objects, f)
        cv2.imwrite( os.path.join(frames_dir, f'{os.path.basename(frame_path)}_objects.png'), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object detection script')
    parser.add_argument('--v3c1_keyframes_dir', required=True, help='Path to the directory containing keyframes')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    args = parser.parse_args()
    inference(args.v3c1_keyframes_dir, args.threshold)
