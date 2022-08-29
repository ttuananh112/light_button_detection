import cv2

from common.inference import InferenceModel
from common.error_detector import ErrorDetector

if __name__ == "__main__":
    img_path = "/home/anhtt163/Desktop/outsource/vid/vid/img/z3383010729342_cc973dacdfaba44bee6a621237d358da.jpg"
    inp = cv2.imread(img_path)

    model = InferenceModel()
    ed = ErrorDetector(ref_buttons=[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    pred = model(inp, save_path="error_detector.png")
    out = ed(pred)
    print(out)
