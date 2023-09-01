from yolov7_package import Yolov7Detector
import cv2

if __name__ == '__main__':
    img = cv2.imread('/Users/iivan/datasets/v3c1_keyframes_selected/00001.mp4/frame_05431.png')
    det = Yolov7Detector(traced=False)
    classes, boxes, scores = det.detect(img)
    img = det.draw_on_image(img, boxes[0], scores[0], classes[0])

    cv2.imshow("image", img)
    cv2.waitKey()