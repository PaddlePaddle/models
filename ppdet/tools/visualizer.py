#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import argparse
import functools
import random
import os
import sys
import json
import cv2

import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from tqdm import tqdm

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
if path not in sys.path:
    sys.path.insert(0, path)
from args import parse_args, print_arguments, add_arguments

__all__ = ['visual_single_img']


def parse_args():
    """ return all args
    """
    parser = argparse.ArgumentParser(
        description='Visualize detection and segmentation')
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('anno_file_path', str, None,
            "path of the COCO style annotation file")
    add_arg('bbox_file_path', str, None,
            "path of detection json file to visualize")
    add_arg('segm_file_path', str, None,
            "path of segmentation json file to visualize")
    add_arg('img_folder', str, None, "path of the source image folder")
    add_arg('output_folder', str, "../../vis_img/",
            "path of the output image folder")
    add_arg('show_border', bool, False, "wether draw the contour of the mask")
    add_arg('thresh', float, 0.5,
            "the score threshold to visualize the bounding boxes")
    args = parser.parse_args()
    return args


def get_color(category_num=80):
    """
    Generate the color list according to the number of categories.

    Args:
        category_num (int): The number of categories.

    Returns:
        color_list (list): the generated color list.
    """
    num_ch = np.power(category_num, 1 / 3.)
    frag = int(np.floor(256 / num_ch))
    color_ch = range(0, 255, frag)
    color_list = []
    for color_r in color_ch:
        for color_g in color_ch:
            for color_b in color_ch:
                color_list.append((color_r, color_g, color_b))
    return color_list


def category_id_to_name(dataset):
    """
    Map the category ID to the name.

    Args:
        dataset (COCO): The dataset instance of the COCO class.
        
    Returns:
        category_name (list): The name of categories ordered by category ID.
    """
    category_name = dict()
    category_ids = dataset.getCatIds()
    categories = [c['name'] for c in dataset.loadCats(category_ids)]
    category_to_id_map = dict(zip(categories, category_ids))
    for iter, cat_name in enumerate(category_to_id_map):
        elem = {}
        elem['name'] = cat_name
        elem['cid'] = category_to_id_map[cat_name]
        elem['phid'] = iter
        elem['orname'] = dataset.loadCats(elem['cid'])[0]['name']
        category_name[elem['cid']] = elem
    return category_name


def visual_single_img(img, bbox_per_img, segm_per_img, img_name, output_folder,
                      color_list, category_name, show_border, thresh):
    """
    Visualize a single image's bounding boxes and segmentations.
    """
    area = []
    scores = []

    # visualize the bounding box
    if len(bbox_per_img) > 0:
        for iter, entry in enumerate(bbox_per_img):
            bbox = np.array(entry['bbox']).astype(np.int32)
            try:
                score = float(entry['score'])
            except:
                score = 1.0
            cat_id = int(entry['category_id'])
            area.append(bbox[2] * bbox[3])
            scores.append(score)
            if score < thresh:
                continue

            cv2.rectangle(img=img,
                          pt1=(bbox[0], bbox[1]),
                          pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          color=color_list[category_name[cat_id]['cid']],
                          thickness=1)
            cv2.putText(img=img,
                        text=category_name[cat_id]['name'],
                        org=(max(bbox[0], 0), max(bbox[1] - 5, 0)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=color_list[category_name[cat_id]['cid']],
                        thickness=1)
    idx_by_area = (np.argsort(-np.array(area)))

    # visualize the segmentation
    if len(segm_per_img) > 0:
        alpha = 0.4
        for iter, idx in enumerate(idx_by_area):
            if scores[idx] < thresh:
                continue
            entry = segm_per_img[idx]
            mask = mask_util.decode(entry['segmentation'])
            img = img.astype(np.float32)
            idx = np.nonzero(mask)
            cat_id = int(entry['category_id'])
            color = color_list[category_name[cat_id]['cid']],

            img[idx[0], idx[1], :] *= 1.0 - alpha
            img[idx[0], idx[1], :] += alpha * np.array(color)

            if show_border:
                contours = cv2.findContours(mask.copy(), cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_NONE)[-2]
                cv2.drawContours(img, contours, -1, _WHITE, border_thick,
                                 cv2.LINE_AA)
            img = img.astype(np.uint8)

    cv2.imwrite(os.path.join(output_folder, img_name), img)


def visual_all_imgs(args, bbox, segm, img_ids, category_name, dataset):
    """
    Visualize all images.
    """
    color_list = get_color(category_num=len(category_name))
    vis_by_img = dict()

    #Initialize the vis_by_img
    for img_id in img_ids:
        vis_by_img[img_id] = dict()
        vis_by_img[img_id]['bbox'] = []
        vis_by_img[img_id]['segm'] = []
    if bbox is not None:
        for iter, entry in enumerate(bbox):
            img_id = entry['image_id']
            vis_by_img[img_id]['bbox'].append(entry)
    if segm is not None:
        for iter, entry in enumerate(segm):
            img_id = entry['image_id']
            vis_by_img[img_id]['segm'].append(entry)

    for iter, img_id in tqdm(enumerate(vis_by_img)):
        bbox_entry = vis_by_img[img_id]['bbox']
        segm_entry = vis_by_img[img_id]['segm']
        img_name = dataset.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(args.img_folder, img_name))

        visual_single_img(img, bbox_entry, segm_entry, img_name, color_list,
                          category_name, args.show_border, args.thresh,
                          args.output_folder)


def load_anno_file(json_file_path):
    """ 
    Load annotation json data from disk.

    Args:
        json_file_path (string): The path of the json file.

    Returns:
        json_data (list):
    """
    assert os.path.splitext(
        json_file_path)[-1][1:] == "json", 'bbox data should be a json file'
    with open(json_file_path, 'rb') as handle:
        json_data = json.load(handle)
        handle.close()
    # If the bbox info is provided by a COCO style annotation file
    if isinstance(json_data, dict):
        json_data = json_data['annotations']
    return json_data


def main():
    args = parse_args()
    print_arguments(args)
    assert args.img_folder is not None, 'The path of the image folder must not be None'
    assert args.anno_file_path is not None, 'Annotation file must be provided for information of categories and images'
    assert os.path.splitext(
        args.anno_file_path
    )[-1][1:] == "json", 'bbox data should be a json file'
    dataset = COCO(args.anno_file_path)
    img_ids = dataset.getImgIds()
    # Keep consistency with the inference phase
    img_ids.sort()
    # Map category ID to the category name
    category_name = category_id_to_name(dataset)

    # Load bbox and segmenation json file from disk
    if args.bbox_file_path is not None:
        bbox = load_anno_file(args.bbox_file_path)
    if args.segm_file_path is not None:
        segm = load_anno_file(args.segm_file_path)

    visual_all_imgs(args, bbox, segm, img_ids, category_name, dataset)


if __name__ == '__main__':
    main()
