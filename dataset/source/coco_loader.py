import numpy as np
import matplotlib
matplotlib.use('Agg')
from pycocotools.coco import COCO

def load(anno_path, sample_num=-1):
    """ Load COCO records with annotations in json file 'anno_path'

    Args:
        @anno_path (str): json file path
        @sample_num (int): number of samples to load, -1 means all

    Returns:
        (records, cname2cid)
        'records' is list of dict whose structure is:
        {
            'im_file': im_fname, # image file name
            'im_id': img_id, # image id
            'h': im_h, # height of image
            'w': im_w, # width
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': gt_poly,
        }
        'cname2cid' is a dict to map category name to class id
    """

    assert anno_path.endswith('.json'), 'invalid coco annotation file[%s]' % (anno_path)
    coco = COCO(anno_path)
    img_ids = coco.getImgIds()
    records = []
    ct = 0

    # mapping category to classid, like:
    #   background:0, first_class:1, second_class:2, ...
    catid2clsid = dict({catid: i + 1 for i, catid in enumerate(coco.getCatIds())})

    cname2cid = dict({coco.loadCats(catid)[0]['name']: clsid for catid,
                                 clsid in catid2clsid.items()})

    for img_id in img_ids:
        img_anno = coco.loadImgs(img_id)[0]
        im_fname = img_anno['file_name']
        im_w = img_anno['width']
        im_h = img_anno['height']

        ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        instances = coco.loadAnns(ins_anno_ids)

        # sanitize bboxes
        valid_instances = []
        for inst in instances:
            x, y, box_w, box_h = inst['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(im_w - 1, x1 + max(0, box_w - 1))
            y2 = min(im_h - 1, y1 + max(0, box_h - 1))
            if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                inst['clean_bbox'] = [x1, y1, x2, y2]
                valid_instances.append(inst)
        num_instance = len(valid_instances)

        gt_bbox = np.zeros((num_instance, 4), dtype=np.float32)
        gt_class = np.zeros((num_instance, ), dtype=np.int32)
        is_crowd = np.zeros((num_instance, ), dtype=np.int32)
        gt_poly = [None] * num_instance

        for i, inst in enumerate(valid_instances):
            catid = inst['category_id']
            gt_class[i] = catid2clsid[catid]
            gt_bbox[i, :] = inst['clean_bbox']
            is_crowd[i] = inst['iscrowd']
            gt_poly[i] = inst['segmentation']

        coco_rec = {
            'im_file': im_fname,
            'im_id': img_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': gt_poly,
            }

        records.append(coco_rec)
        ct += 1
        if sample_num > 0 and ct >= sample_num:
            break

    assert len(records) > 0, 'not found any coco record in %s' % (anno_path)
    return [records, cname2cid]

