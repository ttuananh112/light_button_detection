import torch
import numpy as np
import matplotlib.pyplot as plt

import configs.infer as config_infer
from libs.common.viz import draw_box

from utils.torch_utils import TracedModel
from utils.datasets import letterbox
from utils.general import (non_max_suppression,
                           scale_coords,
                           xyxy2xywh)


class InferenceModel:
    def __init__(self, weight_path):
        self.weight_path = weight_path
        self.device = config_infer.DEVICE
        self.img_size = config_infer.IMG_SIZE
        self.stride = config_infer.STRIDE

        # load model
        self.model = self._load_model()

    def _load_model(self):
        """
        Load model and convert to traced model
        Returns:
        """
        # load checkpoint
        ckpt = torch.load(self.weight_path, map_location=self.device)
        ckpt = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
        # convert to traced model
        model = TracedModel(ckpt, self.device, self.img_size)
        return model

    def _preprocess(self, img0):
        """
        Args:
            img0: raw image np.ndarray (h, w, 3) in RGB
        Returns:
            (torch.Tensor) in shape (1, 3, h, w)
        """
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        # Convert
        img = img.transpose(2, 0, 1)  # (3, h, w)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        # normalize
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # add batch size
        return img

    @staticmethod
    def _postprocess(img, img0, pred):
        """
        Args:
            img: torch.Tensor (1, 3, h, w)
            img0: np.ndarray (h, w, 3)
            pred: torch.Tensor

        Returns:
            (list(list)), for each row: (x, y, w, h)
        """
        det = non_max_suppression(pred, config_infer.CONF_THRESH, config_infer.IOU_THRESH)[0]  # get first batch
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        list_det = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # raw xywh
                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                # line = (cls, *xywh)  # label format
                list_det.append([int(cls), *[int(i) for i in xyxy]])

        return list_det

    def __call__(self, img, show_img=False):
        """
        Run inference
        Args:
            img: np.ndarray (h, w, 3)
        Returns:

        """
        proc_img = self._preprocess(img)  # pre-processed image
        pred = self.model(proc_img)[0]  # get the first batch
        out = self._postprocess(proc_img, img, pred)  # post-process to get output

        if show_img:
            viz = draw_box(out, img)
            plt.imshow(viz)
            plt.axis('off')
            plt.show()
            # plt.savefig('detection.png')
        return out
