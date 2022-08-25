import cv2
import glob
import random
import torch
import numpy as np

from torch.utils.data import Dataset
import classifier.configs as cfg


class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img):
        h, w = img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h, left: left + new_w]
        return img


class Augmentator:
    def __init__(self):
        pass

    def __call__(self, img):
        return img


class ButtonDataset(Dataset):
    def __init__(self, data_folder: str):
        self.list_img = glob.glob(f"{data_folder}/*/*")
        # shuffle
        random.shuffle(self.list_img)
        self.augment = Augmentator()

    def __len__(self):
        return len(self.list_img)

    def _process_inp(self, img_path):
        img = cv2.imread(img_path)[:, :, ::-1]
        # resize to cfg.IMG_SIZE
        img = cv2.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        # apply augmentation
        img = self.augment(img)
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
