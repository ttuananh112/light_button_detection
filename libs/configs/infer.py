class Detection:
    CLASSES = {0: "button"}

    DEVICE = 'cpu'
    IMG_SIZE = 640
    STRIDE = 32

    CONF_THRESH = 0.25
    IOU_THRESH = 0.45
    MODEL_PATH = "libs/yolov7/weights/lb_overfit/best.pt"


class Recognition:
    NUM_CLASS = 7
    CLASSES = {0: "green_on", 1: "green_off",
               2: "red_on", 3: "red_off",
               4: "orange", 5: "white", 6: "unk"}

    DEVICE = 'cpu'
    IMG_SIZE = 32
    MODEL_PATH = "libs/classifier/weights/efficientnetv2_tiny/button-epoch=42-val_f1=0.99.ckpt"
