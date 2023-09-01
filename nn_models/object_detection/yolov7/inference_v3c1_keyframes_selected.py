import argparse
import glob
import json
import os

from yolov7_package import Yolov7Detector
import cv2


def inference(v3c1_keyframes_dir: str, threshold: float):
    det = Yolov7Detector(traced=False)
    img_folder_list = [os.path.join(v3c1_keyframes_dir, img_folder) for img_folder in os.listdir(v3c1_keyframes_dir)]
    for img_folder in img_folder_list:
        img_path = glob.glob(f'{img_folder}/*.png')[0]
        img = cv2.imread(img_path)
        classes, boxes, scores = det.detect(img)
        objects = []
        for class_id, box, score in zip(classes[0], boxes[0], scores[0]):
            if score >= threshold:
                objects.append({det.names[class_id]: box})
        store_path = os.path.join(img_folder, 'objects.json')
        if os.path.isfile(store_path):
            os.remove(store_path)
        with open(store_path, 'w') as f:
            json.dump(objects, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object detection script')
    parser.add_argument('--v3c1_keyframes_dir', required=True, help='Path to the directory containing keyframes')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    args = parser.parse_args()
    inference(args.v3c1_keyframes_dir, args.threshold)
