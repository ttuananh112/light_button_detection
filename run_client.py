import cv2
import json
import argparse

import configs.conn as conn
import libs.connection.client as client
from libs.common.viz import draw_box

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Image's path", required=True)
    args = parser.parse_args()

    addr = f'http://localhost:{conn.PORT}{conn.URL}'
    img = cv2.imread(args.image)
    response = client.send_image(addr, img)

    # decode response
    decoded_res = json.loads(response.text)
    detection = decoded_res["detection"]
    print(*detection, sep="\n")

    img_box = draw_box(detection, img)
    cv2.imwrite("detection.png", img_box)
