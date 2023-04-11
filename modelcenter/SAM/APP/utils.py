import os
import time
import numpy as np
import cv2
from PIL import Image

CACHE_DIR = ".temp"


def delete_result():
    """clear old result in `.temp`"""
    results = sorted(os.listdir(CACHE_DIR))
    for res in results:
        if int(time.time()) - int(os.path.splitext(res)[0]) > 10000:
            os.remove(os.path.join(CACHE_DIR, res))


def download(img):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    while True:
        name = str(int(time.time()))
        tmp_name = os.path.join(CACHE_DIR, name + '.jpg')
        if not os.path.exists(tmp_name):
            break
        else:
            time.sleep(1)

    img.save(tmp_name, 'png')
    return tmp_name


def get_pseudo_color_map(pred, color_map=None):
    """
    Get the pseudo color image.

    Args:
        pred (numpy.ndarray): the origin predicted image.
        color_map (list, optional): the palette color map. Default: None,
            use paddleseg's default color map.

    Returns:
        (numpy.ndarray): the pseduo image.
    """
    pred_mask = Image.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


def masks2pseudomap(masks):
    result = np.ones(masks[0]["segmentation"].shape, dtype=np.uint8) * 255
    for i, mask_data in enumerate(masks):
        result[mask_data["segmentation"] == 1] = i + 1
    pred_result = result
    result = get_pseudo_color_map(result)

    return pred_result, result


def visualize(image, result, color_map, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c3, c2, c1))

    vis_result = cv2.addWeighted(image, weight, pseudo_img, 1 - weight, 0)
    return vis_result
