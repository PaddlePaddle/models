import argparse
import os
import pickle as pkl
import numpy as np
import matplotlib
matplotlib.use('Agg')

from pycocotools.coco import COCO

dataset_split_mapping = {
    "train2014": "train2014",
    "val2014": "val2014",
    "valminusminival2014": "val2014",
    "minival2014": "val2014",
    "train2017": "train2017",
    "val2017": "val2017",
    "test-dev2017": "test2017"
}


def parse_args():
    """ parse arguments
    """
    parser = argparse.ArgumentParser( \
        description='Generate SimpleDet GroundTruth Database')

    parser.add_argument('--dataset', help='dataset name', type=str)
    parser.add_argument('--dataset-split', type=str, \
        help='dataset split, e.g. train2017, minival2014')
    parser.add_argument('--save-dir', type=str, \
        help='directory to save roidb files', default='data/tests')
    parser.add_argument('--samples', default=-1, type=int, \
        help='number of samples to dump,default to all')

    args = parser.parse_args()
    return args


def generate_groundtruth_database(\
    dataset_name, dataset_split, samples=-1):
    """ generate groud truth db which
    """

    data_root = "data/%s" % (dataset_name)
    annotation_type = 'image_info' if 'test' in dataset_split else 'instances'
    annotation_path = "annotations/%s_%s.json" % (annotation_type, dataset_split)
    annotation_path = os.path.join(data_root, annotation_path)

    dataset = COCO(annotation_path)
    img_ids = dataset.getImgIds()
    roidb = []
    ct = 0
    for img_id in img_ids:
        img_anno = dataset.loadImgs(img_id)[0]
        im_filename = img_anno['file_name']
        im_w = img_anno['width']
        im_h = img_anno['height']

        ins_anno_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=False)
        trainid_to_datasetid = dict({i + 1: cid for i, cid in enumerate(dataset.getCatIds())})  # 0 for bg
        datasetid_to_trainid = dict({cid: tid for tid, cid in trainid_to_datasetid.items()})
        instances = dataset.loadAnns(ins_anno_ids)

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
        gt_poly = [None] * num_instance

        for i, inst in enumerate(valid_instances):
            cls = datasetid_to_trainid[inst['category_id']]
            gt_bbox[i, :] = inst['clean_bbox']
            gt_class[i] = cls
            gt_poly[i] = inst['segmentation']

        split = dataset_split_mapping[dataset_split]
        roi_rec = {
            'image_url': '%s/%s' % (split, im_filename),
            'im_id': img_id,
            'h': im_h,
            'w': im_w,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': gt_poly,
            'flipped': False}

        roidb.append(roi_rec)
        ct += 1
        if samples > 0 and ct >= samples:
            break

    return roidb


if __name__ == "__main__":
    """ make sure your data is stored in 'data/${args.dataset}'

    usage:
        python generate_roidb.py --dataset COCO17 \
        --dataset-split val2017 --samples 100 --save-dir data/tests
    """

    args = parse_args()

    dsname = args.dataset
    dsplit = args.dataset_split
    samples = args.samples 
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    roidb = generate_groundtruth_database(dsname, dsplit, samples)
    samples = len(roidb)
    roidb_fname = save_dir + "/%s_%s.roidb" % (dsname, dsplit)
    with open(roidb_fname, "wb") as fout:
        pkl.dump(roidb, fout)

    print('dumped %d samples to file[%s]' % (samples, roidb_fname))

