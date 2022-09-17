import cv2
import jsonpickle
import numpy as np
from flask import Flask, request, Response

import libs.configs.conn as conn
import libs.configs.mapping as mapping
import libs.configs.message as message

from libs.common.inference import InferenceModel
from libs.common.error_detector import ErrorDetector

# Initialize the Flask application
app = Flask(__name__)
# Load model
model = InferenceModel()
error_detection = ErrorDetector()


# route http posts to this method
@app.route(conn.URL, methods=['POST'])
def detect():
    """
    Detection API
    - Request into IP:PORT/URL/?position={pos}
    with body = encoded image by cv2
    - Response:
    {
        "message":... , # msg
        "is_false_detection":... ,  # error in detection module
        "is_error":... ,  # get error-button
        "detection":...  # bbox detection
    }
    Returns:

    """
    # convert string of image data to uint8
    mapping_index = request.args.get(message.POSITION, type=int)
    nparr = np.fromstring(request.data, np.uint8)

    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # image should be in RGB format
    # get reference buttons
    ref_buttons = mapping.REF_BUTTONS[mapping_index]
    error_detection.set_ref_buttons(ref_buttons)

    # inference
    pred = model(img)
    error = error_detection(pred)

    # build a response dict to send back to client
    response = {message.DETECTION: pred,
                **error}

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(
        response=response_pickled,
        status=message.RESPONSE_OK,
        mimetype=message.MIMETYPE
    )
