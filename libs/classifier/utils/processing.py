import os
import cv2
import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

from utils.general import xywhn2xyxy


class DataProcessor:
    def __init__(self, save_folder):
        self.save_root_folder = save_folder
        # create folder
        if not os.path.exists(self.save_root_folder):
            os.makedirs(self.save_root_folder)
        # classes_counter
        self.counter = {}

    def process_folder(self, folder_path: str):
        list_img = sorted(glob.glob(f"{folder_path}/images/*"))
        list_lbl = sorted(glob.glob(f"{folder_path}/labels/*"))

        for img_path, lbl_path in tqdm(zip(list_img, list_lbl),
                                       total=len(list_lbl)):
            np_img = cv2.imread(img_path)
            np_lbl = self.get_list_bboxes(lbl_path).to_numpy()

            h, w = np_img.shape[:2]
            xyxys = xywhn2xyxy(np_lbl[:, 1:], w, h)
            cls = np_lbl[:, 0].reshape(-1, 1)
            cls_xyxys = np.concatenate([cls, xyxys], axis=-1)
            for cls, roi in self.crop_roi(np_img, cls_xyxys):
                cls_folder = f"{self.save_root_folder}/{cls}"
                if not os.path.exists(cls_folder):
                    os.makedirs(cls_folder)

                # update counter
                self.counter[cls] = 0 if cls not in self.counter else self.counter[cls] + 1
                roi_path = f"{cls_folder}/{self.counter[cls]:08d}.png"
                cv2.imwrite(roi_path, roi)

    @staticmethod
    def get_list_bboxes(lbl_path):
        df_bbox = pd.read_csv(lbl_path, sep=" ", header=None)
        return df_bbox

    @staticmethod
    def crop_roi(img, cls_xyxys):
        for cls_xyxy in cls_xyxys:
            cls, x0, y0, x1, y1 = cls_xyxy
            cropped = img[int(y0): int(y1), int(x0): int(x1), :]
            yield int(cls), cropped
