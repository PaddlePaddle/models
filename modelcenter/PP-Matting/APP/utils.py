import os
import time

from collections import OrderedDict
import numpy as np
import pymatting
import cv2
from PIL import Image

SIZES = OrderedDict({
    "1 inch": {
        'physics': (25, 35),
        'pixels': (295, 413)
    },
    "1 inch smaller": {
        'physics': (22, 32),
        'pixels': (260, 378)
    },
    "1 inch larger": {
        'physics': (33, 48),
        'pixels': (390, 567)
    },
    "2 inches": {
        'physics': (35, 49),
        'pixels': (413, 579)
    },
    "2 inches smaller": {
        'physics': (35, 45),
        'pixels': (413, 531)
    },
    "2 inches larger": {
        'physics': (35, 53),
        'pixels': (413, 626)
    },
    "3 inches": {
        'physics': (55, 84),
        'pixels': (649, 991)
    },
    "4 inches": {
        'physics': (76, 102),
        'pixels': (898, 1205)
    },
    "5 inches": {
        'physics': (89, 127),
        'pixels': (1050, 1500)
    }
})

# R, G, B
COLOR_MAP = {
    'White': [255, 255, 255],
    'Blue': [0, 191, 243],
    'Red': [255, 0, 0]
}

# jpg compress ratio
SAVE_SIZE = {'Small': 50, 'Middle': 75, 'Large': 95}


def delete_result():
    """clear old result in `.temp`"""
    root = '.temp'
    results = sorted(os.listdir(root))
    for res in results:
        if int(time.time()) - int(os.path.splitext(res)[0]) > 900:
            os.remove(os.path.join(root, res))


def clear_all():
    delete_result()
    return None, None, size_play()[0], 'White', 'Large', None


def size_play():
    sizes = []
    for k, v in SIZES.items():
        size = ''.join([
            k, '(', str(v['physics'][0]), 'x', str(v['physics'][1]), 'mm,',
            str(v['pixels'][0]), 'x', str(v['pixels'][1]), 'px)'
        ])
        sizes.append(size)
    return sizes


def bg_replace(img, alpha, bg_name):
    bg = COLOR_MAP[bg_name]
    bg = np.array(bg)[None, None, :]
    alpha = alpha / 255.
    pymatting.estimate_foreground_ml(img / 255., alpha) * 255
    alpha = alpha[:, :, None]
    res = alpha * img + (1 - alpha) * bg
    return res.astype('uint8')


def adjust_size(img, size_index):
    key = list(SIZES.keys())[size_index]
    w_o, h_o = SIZES[key]['pixels']

    # scale
    h_ori, w_ori = img.shape[:2]
    scale = max(w_o / w_ori, h_o / h_ori)
    if scale > 1:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA
    img_scale = cv2.resize(
        img, dsize=None, fx=scale, fy=scale, interpolation=interpolation)

    # crop
    h_scale, w_scale = img_scale.shape[:2]
    h_cen = h_scale // 2
    w_cen = w_scale // 2
    h_start = max(0, h_cen - h_o // 2)
    h_end = min(h_scale, h_start + h_o)
    w_start = max(0, w_cen - w_o // 2)
    w_end = min(w_scale, w_start + w_o)
    img_c = img_scale[h_start:h_end, w_start:w_end]

    return img_c


def download(img, size):
    q = SAVE_SIZE[size]
    while True:
        name = str(int(time.time()))
        tmp_name = './.temp/' + name + '.jpg'
        if not os.path.exists(tmp_name):
            break
        else:
            time.sleep(1)
    dir_name = os.path.dirname(tmp_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    im = Image.fromarray(img)
    im.save(tmp_name, 'jpeg', quality=q, dpi=(300, 300))
    return tmp_name
