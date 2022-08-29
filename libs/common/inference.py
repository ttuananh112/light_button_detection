import abc
import cv2
import torch
import numpy as np

from typing import List, Union
import libs.configs.infer as config_infer
from libs.common.viz import draw_box

from utils.torch_utils import TracedModel
from utils.datasets import letterbox
from utils.general import (non_max_suppression,
                           scale_coords)

# from libs.classifier.model.button_classifier import ButtonClassifier
from libs.classifier.model.efficientnetv2 import effnetv2_tiny


class BaseModel:
    @abc.abstractmethod
    def _load_model(self):
        pass

    @abc.abstractmethod
    def _preprocess(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _postprocess(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class DetectionModel(BaseModel):
    def __init__(self):
        self.device = config_infer.Detection.DEVICE
        self.img_size = config_infer.Detection.IMG_SIZE
        self.stride = config_infer.Detection.STRIDE

        # load model
        self.model = self._load_model()

    def _load_model(self) -> TracedModel:
        """
        Load model and convert to traced model
        Returns:
        """
        # load checkpoint
        ckpt = torch.load(config_infer.Detection.MODEL_PATH, map_location=self.device)
        ckpt = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
        # convert to traced model
        model = TracedModel(ckpt, self.device, self.img_size)
        return model

    def _preprocess(
            self,
            img0: Union[str, np.ndarray]
    ) -> torch.Tensor:
        """
        Args:
            img0: raw image np.ndarray (h, w, 3) in BGR
        Returns:
            (torch.Tensor) in shape (1, 3, h, w)
        """
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        # Convert to RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # (3, h, w)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        # normalize
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # add batch size
        return img

    def _postprocess(
            self,
            img: torch.Tensor,
            img0: np.ndarray,
            pred: torch.Tensor
    ) -> List[List]:
        """
        Args:
            img: torch.Tensor (1, 3, h, w)
            img0: np.ndarray (h, w, 3)
            pred: torch.Tensor

        Returns:
            (list(list)), for each row: (x, y, w, h)
        """
        det = non_max_suppression(pred,
                                  config_infer.Detection.CONF_THRESH,
                                  config_infer.Detection.IOU_THRESH)[0]  # get first batch
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

    def __call__(
            self,
            img: np.ndarray,
            save_path: str = None
    ):
        """
        Run inference
        Args:
            img: np.ndarray (h, w, 3) in BGR
        Returns:

        """
        proc_img = self._preprocess(img)  # pre-processed image
        pred = self.model(proc_img)[0]  # get the first batch
        out = self._postprocess(proc_img, img, pred)  # post-process to get output

        if save_path:
            viz = draw_box(out, img,
                           label_mapping=config_infer.Detection.CLASSES)
            cv2.imwrite(save_path, viz)
        return out


class RecognitionModel(BaseModel):
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        # model = ButtonClassifier()
        model = effnetv2_tiny(num_classes=config_infer.Recognition.NUM_CLASS)
        model.load_state_dict(
            torch.load(config_infer.Recognition.MODEL_PATH)["state_dict"]
        )
        model.eval().to(config_infer.Recognition.DEVICE)
        return model

    def _preprocess(self, img: Union[str, np.ndarray]):
        img = img[:, :, ::-1]
        # resize to cfg.IMG_SIZE
        img = cv2.resize(img, (config_infer.Recognition.IMG_SIZE,) * 2)
        # convert to torch.float
        img = torch.from_numpy(img.copy()).float()
        # to shape (1, c, h, w)
        img = torch.permute(img, (2, 0, 1)).unsqueeze(0).to(config_infer.Recognition.DEVICE)
        # normalize
        img /= 255.
        return img

    def _postprocess(self, pred: torch.Tensor):
        max_idx = torch.argmax(pred)
        return int(max_idx)

    def __call__(self, img: np.ndarray):
        proc_img = self._preprocess(img)  # pre-processed image
        pred = self.model(proc_img)[0]  # get the first batch
        out = self._postprocess(pred)  # post-process to get output
        return out


class InferenceModel:
    def __init__(self):
        self.detection = DetectionModel()
        self.recognition = RecognitionModel()

    def __call__(
            self,
            img: np.ndarray,
            save_path: str = None
    ):
        img0 = np.copy(img)
        out_det = self.detection(img)
        for i, det in enumerate(out_det):
            cls, x0, y0, x1, y1 = det
            cropped = img0[y0: y1, x0: x1, :]
            out_reg = self.recognition(cropped)
            out_det[i][0] = out_reg

        if save_path is not None:
            viz = draw_box(out_det, img0,
                           label_mapping=config_infer.Recognition.CLASSES)
            cv2.imwrite(save_path, viz)
        return out_det
