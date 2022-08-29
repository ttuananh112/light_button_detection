import os

import cv2
import argparse
import numpy as np

import libs.configs.infer as config_infer
from libs.common.viz import draw_box
from libs.common.inference import InferenceModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Image's or video's path", required=True)
    parser.add_argument("-o", "--output", type=str, help="Folder destination", required=True)
    args = parser.parse_args()

    # output preparation
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # setup output
    fname = os.path.basename(args.input)
    out_path = f"{args.output}/det_{fname}"

    # init model
    model = InferenceModel()
    # for video
    if args.input.split(".")[-1] == "mp4":
        # read the video
        video = cv2.VideoCapture(args.input)

        fps = video.get(cv2.CAP_PROP_FPS)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(out_path,
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              fps, size)

        while True:
            ret, img0 = video.read()
            if ret:
                img = np.copy(img0)
                detection = model(img)
                # visualize
                viz = draw_box(detection, img,
                               label_mapping=config_infer.Recognition.CLASSES)
                out.write(viz)
            else:
                break
        # clear streaming
        video.release()
        out.release()

    # it should be images
    else:
        # read image
        img0 = cv2.imread(args.input)
        img = np.copy(img0)

        detection = model(img)
        # visualize
        viz = draw_box(detection, img,
                       label_mapping=config_infer.Recognition.CLASSES)
        cv2.imwrite(out_path, viz)
