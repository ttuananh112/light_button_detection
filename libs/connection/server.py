import cv2
import jsonpickle
import numpy as np
from flask import Flask, request, Response

import configs.conn as conn
import configs.infer as config_infer
from libs.common.inference import InferenceModel

# Initialize the Flask application
app = Flask(__name__)
# Load model
model = InferenceModel(config_infer.MODEL_PATH)


# route http posts to this method
@app.route(conn.URL, methods=['POST'])
def detect():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)

    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # image should be in RGB format

    # inference
    pred = model(img)

    # build a response dict to send back to client
    response = {"message": "received",
                "detection": pred}

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")
