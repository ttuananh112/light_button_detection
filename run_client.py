import cv2
import json
import configs.conn as conn
import libs.connection.client as client

from libs.common.viz import draw_box

if __name__ == "__main__":
    addr = f'http://localhost:{conn.PORT}{conn.URL}'

    img_path = "/home/anhtt163/PycharmProjects/outsource/dataset/phone/LightButton.v2i.yolov7pytorch/train/images/3383010899951_mp4-14_jpg.rf.9d0c9d552cbe5d205d21e3f2f8ae8091.jpg"
    img = cv2.imread(img_path)

    response = client.send_image(addr, img)

    # decode response
    decoded_res = json.loads(response.text)
    detection = decoded_res["detection"]
    print(*detection, sep="\n")

    img_box = draw_box(detection, img)
    cv2.imwrite("detection.png", img_box)
