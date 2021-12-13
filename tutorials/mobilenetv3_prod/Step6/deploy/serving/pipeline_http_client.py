import numpy as np
import requests
import json
import cv2
import base64
import os


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='Paddle Serving', add_help=add_help)

    parser.add_argument('--img-path', default="../../images/demo.jpg")
    args = parser.parse_args()
    return args


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')


def main(args):
    url = "http://127.0.0.1:18080/mobilenet_v3_small/prediction"
    logid = 10000

    img_path = args.img_path
    with open(img_path, 'rb') as file:
        image_data1 = file.read()
    image = cv2_to_base64(image_data1)
    data = {"key": ["image"], "value": [image], "logid": logid}
    r = requests.post(url=url, data=json.dumps(data))
    print(r.json())


if __name__ == "__main__":
    args = get_args()
    main(args)
