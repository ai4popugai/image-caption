import argparse
import glob
import json
import os

from yolov7_package import Yolov7Detector
import cv2

from db import SQLiteDb


def yolo_inference(frames_dir: str, threshold: float = 0.5, database: SQLiteDb = None,):
    det = Yolov7Detector(traced=False)
    video_id = os.path.basename(frames_dir)

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
        if database is None:
            store_path = os.path.join(frames_dir, f'{os.path.basename(frame_path)}_objects.json')
            with open(store_path, 'w') as f:
                json.dump(objects, f)
            cv2.imwrite(os.path.join(frames_dir, f'{os.path.basename(frame_path)}_objects.png'), img)
        else:
            keyframe_id = os.path.basename(frame_path)
            database.add_objects_to_row(video_id, keyframe_id, objects)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object detection script')
    parser.add_argument('--frames_dir', required=True, help='Path to the directory containing keyframes')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    args = parser.parse_args()
    yolo_inference(args.frames_dir, args.threshold)
