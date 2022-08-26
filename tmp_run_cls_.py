import cv2
from common.inference import InferenceModel
from common.timeit import TimeIt


if __name__ == "__main__":
    img_path = "/home/anhtt163/Desktop/outsource/vid/vid/img/z3383010729342_cc973dacdfaba44bee6a621237d358da.jpg"
    inp = cv2.imread(img_path)
    model = InferenceModel()
    with TimeIt("Recog"):
        out = model(inp, show_img=True)
