import cv2
import glob
import random
import torch
import numpy as np
from typing import List

from torch.utils.data import Dataset
import libs.configs.infer as config_infer


class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        pass

    def __call__(self, img, min_ratio=2 / 3):
        h, w = img.shape[:2]
        new_h = np.random.randint(int(min_ratio * h), h)
        new_w = np.random.randint(int(min_ratio * w), w)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h, left: left + new_w]
        return img


class BrightnessAdjustment:
    def __init__(self, value: int = 20):
        self.value = value

    def __call__(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # cast to int16 to calculate with minus number
        v = v.astype(np.int16)
        # adapt value
        value = np.random.randint(-20, 20)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = np.stack((h, s, v), axis=-1).astype(np.uint8)
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img


class Augmentator:
    def __init__(
            self,
            list_aug: List,
            percentage: float = 0.5
    ):
        self.list_aug = list_aug
        self.percentage = percentage

    def __call__(self, img):
        for aug in self.list_aug:
            if np.random.rand() > self.percentage:
                img = aug(img)
        return img


class ButtonDataset(Dataset):
    def __init__(self, data_folder: str):
        self.list_img = glob.glob(f"{data_folder}/*/*")
        # shuffle
        random.shuffle(self.list_img)
        self.augment = Augmentator([RandomCrop(), BrightnessAdjustment()])

    def __len__(self):
        return len(self.list_img)

    def _process_inp(self, img_path):
        img = cv2.imread(img_path)[:, :, ::-1]  # RGB
        # apply augmentation
        img = self.augment(img)
        # resize to cfg.IMG_SIZE
        img = cv2.resize(img, (config_infer.Recognition.IMG_SIZE,) * 2)
        # convert to torch.float
        img = torch.from_numpy(img.copy()).float()
        # to shape (c, h, w)
        img = torch.permute(img, (2, 0, 1))
        # normalize
        img /= 255.
        return img

    @staticmethod
    def _process_out(img_path):
        # get label from folder name
        lbl = int(img_path.split("/")[-2])
        # # one-hot
        # one_hot = torch.zeros(cfg.NUM_CLASS)
        # one_hot[lbl] = 1.
        # return one_hot
        return lbl

    def __getitem__(self, idx):
        img_path = self.list_img[idx]
        img = self._process_inp(img_path)
        lbl = self._process_out(img_path)
        return img, lbl


if __name__ == "__main__":
    ds = ButtonDataset("/home/anhtt163/PycharmProjects/outsource/dataset/phone/classifier/data")
    for im, lb in ds:
        print(im.shape)
        print(lb)
        break
