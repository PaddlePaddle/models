# Ref: https://github.com/waspinator/pycococreator/blob/master/examples/shapes/shapes_to_coco.py
import datetime
import os
import re
import fnmatch
import cv2
import numpy as np

INFO = {
    "description": "AGE Challenge Location",
    "url": "https://age.grand-challenge.org/PaddlePaddle/",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "shangfangxin@baidu.com",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'point',
        'supercategory': 'shape',
    },
]

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[1],
            "height": image_size[0],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

def create_annotation_info(image, annotation_id, image_id, category_info, bounding_box=None):

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": category_info["is_crowd"],
        "area": bounding_box[2] * bounding_box[3],
        "bbox": bounding_box,
        "segmentation": [[]],
        "width": image.shape[1],
        "height": image.shape[0],
    } 

    return annotation_info

def create_anno_info(image, point_x, point_y, image_id, category_info, segmentation_id, box_range):
    bounding_box = [point_x - box_range, point_y - box_range, box_range*2, box_range*2]
    return create_annotation_info(image,
        segmentation_id, image_id, category_info, bounding_box)


def get_coco_dict(img_path, data_list, box_range=20):

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    for item in data_list:
        image_filename, p_x, p_y = item
        p_x, p_y = int(float(p_x)), int(float(p_y))
        image_filename = os.path.join(img_path, image_filename)

        image = cv2.imread(image_filename)
        image_info = create_image_info(
            image_id, os.path.basename(image_filename), image.shape)
        coco_output["images"].append(image_info)

        # filter for associated png annotations
        class_id = 1
        category_info = {'id': class_id, 'is_crowd': 0}

        if p_x != -1 and p_y != -1:
            coco_output["annotations"].append(
                create_anno_info(image, p_x, p_y, image_id, category_info, segmentation_id, box_range))
            segmentation_id = segmentation_id + 1

        image_id = image_id + 1
    return coco_output