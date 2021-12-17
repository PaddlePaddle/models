import numpy as np
import requests
import json
import cv2
import base64
import os


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')


if __name__ == "__main__":
    url = "http://127.0.0.1:18080/tipc_example/prediction"
    logid = 10000
    img_path = "../../images/demo.jpg"
    with open(img_path, 'rb') as file:
        image_data = file.read()
    # data should be transformed to the base64 format
    image = cv2_to_base64(image_data)
    data = {"key": ["image"], "value": [image], "logid": logid}
    # send requests
    r = requests.post(url=url, data=json.dumps(data))
    print(r.json())
