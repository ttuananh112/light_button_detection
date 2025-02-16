import cv2
import json
import argparse

import libs.configs.conn as conn
import libs.configs.infer as config_infer
import libs.connection.client as client
from libs.common.viz import draw_box
from libs.common.timeit import TimeIt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Image's path", required=True)
    args = parser.parse_args()
    # read image
    img = cv2.imread(args.image)
    with TimeIt("Response"):
        addr = f'http://localhost:{conn.PORT}{conn.URL}'
        response = client.send_image(addr, img)
        # decode response
        decoded_res = json.loads(response.text)
        detection = decoded_res["detection"]
        print(json.dumps(decoded_res, sort_keys=True, indent=4))

    # visualize
    viz = draw_box(detection, img,
                   label_mapping=config_infer.Recognition.CLASSES)
    cv2.imwrite('response.png', viz)
