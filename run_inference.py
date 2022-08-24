import cv2
from libs.common.inference import InferenceModel
from libs.common.timeit import TimeIt

if __name__ == "__main__":
    with TimeIt("Load model"):
        model_path = "/home/anhtt163/PycharmProjects/light_button_detection/weights/lb5/best.pt"
        model = InferenceModel(model_path)

    with TimeIt("Load image"):
        img = cv2.imread("/home/anhtt163/PycharmProjects/outsource/dataset/phone/1class/train/images/3383010899951_mp4-1_jpg.rf.d95055ba82a3b68f532bb24be3f7d695.jpg")
        img = img[:, :, ::-1]  # BGR to RGB

    with TimeIt("Inference"):
        # print(img.shape)
        pred = model(img, show_img=True)
        # print(pred)
