import cv2
import requests
import numpy as np
from typing import Union


def send_image(
        url: str,
        img: Union[np.ndarray, str]
):
    """
    Example to send request to server...
    Args:
        url: (str)
        img: [np.ndarray, str]

    Returns:

    """
    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    if isinstance(img, str):
        img = cv2.imread(img)
    elif not isinstance(img, np.ndarray):
        return {"message": "wrong image's type"}

    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)

    # add mapping
    url += "?position=0"

    # send http request with image and receive response
    res = requests.post(
        url=url,
        data=img_encoded.tobytes(),
        headers=headers
    )
    return res
