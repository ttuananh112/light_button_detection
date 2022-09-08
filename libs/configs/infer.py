class Detection:
    CLASSES = {0: "button"}

    DEVICE = 'cuda:0'  # 'cpu'
    IMG_SIZE = 640
    STRIDE = 32

    CONF_THRESH = 0.25
    IOU_THRESH = 0.45
    MODEL_PATH = "libs/yolov7/weights/lb_1class_poc/best.pt"


class Recognition:
    NUM_CLASS = 10
    CLASSES = {0: "green_on", 1: "green_off",
               2: "red_on", 3: "red_off",
               4: 'orange_on', 5: 'orange_off',
               6: 'blue_on', 7: 'blue_off',
               8: 'white', 9: 'unknown'}

    DEVICE = 'cuda:0'  # 'cpu'
    IMG_SIZE = 32
    MODEL_PATH = "libs/classifier/weights/efficientnetv2_tiny_poc/button-epoch=18-val_f1=0.99.ckpt"

    ON_CLASSES = [i for i, v in enumerate(CLASSES.values())
                  if "_on" in v]
