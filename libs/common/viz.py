import cv2
import numpy as np


def draw_box(xyxys, img0):
    img = np.copy(img0).astype(np.uint8)
    for xyxy in xyxys:
        cls = int(xyxy[0])
        start_point = [int(i) for i in xyxy[1:3]]
        end_point = [int(i) for i in xyxy[3:]]
        img = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
        img = cv2.putText(img, f"{cls}", start_point,
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, 2)
    return img
