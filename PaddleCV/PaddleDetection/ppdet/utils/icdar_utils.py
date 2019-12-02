import os
import numpy as np
import logging
logger = logging.getLogger(__name__)


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    if use_default_label or anno_file is None \
            or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "wider-face categories.".format(anno_file))
        return icdar_category_info(with_background)
    else:
        logger.info("Load categories from {}".format(anno_file))
        return get_category_info_from_anno(anno_file, with_background)


def icdar_label(with_background=True):
    labels_map = {'text': 1}
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map


def icdar_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of mixup wider_face dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    label_map = icdar_label(with_background)
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    if with_background:
        cats.insert(0, 'background')

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name


def get_category_info_from_anno(anno_file, with_background=True):
    """
    Get class id to category id map and category id
    to category name map from annotation file.
    Args:
        anno_file (str): annotation file path
        with_background (bool, default True):
            whether load background as class 0.
    """
    cats = []
    with open(anno_file) as f:
        for line in f.readlines():
            cats.append(line.strip())

    if cats[0] != 'background' and with_background:
        cats.insert(0, 'background')
    if cats[0] == 'background' and not with_background:
        cats = cats[1:]

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name


def bbox2out(results, clsid2catid, is_bbox_normalized=False):
    """
    Args:
        results: request a dict, should include: `bbox`, `im_id`,
                 if is_bbox_normalized=True, also need `im_shape`.
        clsid2catid: class id to category id map of COCO2017 dataset.
        is_bbox_normalized: whether or not bbox is normalized.
    """
    final_result = []
    for t in results:
        bboxes = t['bbox'][0]
        #lengths = t['bbox'][1][0]
        lengths = [bboxes.shape[0]]
        im_info = t['im_info'][0]
        im_ids = np.array(t['im_id'][0])
        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i][0])
            for j in range(num):
                dt = bboxes[k]
                dt[2:] /= im_info[0][2]
                clsid, score, x1, y1, x2, y2, x3, y3, x4, y4 = dt.tolist()
                catid = (clsid2catid[int(clsid)])
                bbox = [x1, y1, x2, y2, x3, y3, x4, y4]
                icdar_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': score,
                }
                final_result.append(icdar_res)
                k += 1
    return final_result
