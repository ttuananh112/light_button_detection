import cv2
import numpy as np

from typing import List
import configs.infer as config_infer


def draw_box(
        xyxys: List[List],
        img0: np.ndarray
) -> np.ndarray:
    """

    Args:
        xyxys: List[List] in shape [class, tl_x, tl_y, br_x, br_y]
        img0:

    Returns:

    """
    img = np.copy(img0).astype(np.uint8)
    for xyxy in xyxys:
        cls = int(xyxy[0])
        start_point = [int(i) for i in xyxy[1:3]]
        end_point = [int(i) for i in xyxy[3:]]
        img = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 1)

        start_point[1] -= 5  # lift-up start point
        img = cv2.putText(img, config_infer.CLASSES[cls], start_point,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, 1)
    return img
