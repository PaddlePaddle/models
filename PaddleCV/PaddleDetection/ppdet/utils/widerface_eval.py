# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
from PIL import Image

from ..data.source.widerface_loader import widerface_label
from .coco_eval import bbox2out

import logging
logger = logging.getLogger(__name__)

__all__ = ['WiderfaceEval', 'bbox2out', 'get_category_info']


class WiderfaceEval(object):
    """
    TODO(yuguanghua): add comments.
    """

    def __init__(self,
                 exe,
                 compile_program,
                 fetches,
                 reader,
                 anno_file,
                 pred_dir='pred',
                 eval_mode='matlab'):
        self.exe = exe
        self.compile_program = compile_program
        self.fetches = fetches
        self.reader = reader
        self.anno_file = anno_file
        self.pred_dir = pred_dir
        self.eval_mode = eval_mode
        assert eval_mode in ['matlab', 'python']
        # start eval
        self.bbox_eval()

    def bbox_eval(self):
        imid2path = self.reader.imid2path
        dets_list = []
        for iter_id, image_path in imid2path.iteritems():
            image = Image.open(image_path).convert('RGB')
            shrink, max_shrink = get_shrink(image.size[1], image.size[0])

            det0 = self.detect_face(image, shrink)
            det1 = self.flip_test(image, shrink)
            [det2, det3] = self.multi_scale_test(image, max_shrink)
            det4 = self.multi_scale_test_pyramid(image, max_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)
            if self.eval_mode == 'matlab':
                save_widerface_bboxes_matlab(image_path, dets, self.pred_dir)
            else:
                dets_list.append(dets)
            logger.info('Test iter {}'.format(iter_id))
        if self.eval_mode == 'python':
            pred_res_file = save_widerface_bboxes_py(imid2path, dets_list,
                                                     self.pred_dir)
            calculate_ap_py(pred_res_file, self.anno_file, self.pred_dir)
        logger.info("Finish evaluation.")

    def detect_face(self, image, shrink):
        image_shape = [3, image.size[1], image.size[0]]
        if shrink != 1:
            h, w = int(image_shape[1] * shrink), int(image_shape[2] * shrink)
            image = image.resize((w, h), Image.ANTIALIAS)
            image_shape = [3, h, w]

        img = np.array(image)
        img = to_chw_bgr(img)
        mean = [104., 117., 123.]
        scale = 0.007843
        img = img.astype('float32')
        img -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
        img = img * scale
        img = [img]
        img = np.array(img)
        detection, = self.exe.run(self.compile_program,
                                  feed={'image': img},
                                  fetch_list=[self.fetches['bbox']],
                                  return_numpy=False)
        detection = np.array(detection)
        # layout: xmin, ymin, xmax. ymax, score
        if np.prod(detection.shape) == 1:
            logger.info("No face detected")
            return np.array([[0, 0, 0, 0, 0]])
        det_conf = detection[:, 1]
        det_xmin = image_shape[2] * detection[:, 2] / shrink
        det_ymin = image_shape[1] * detection[:, 3] / shrink
        det_xmax = image_shape[2] * detection[:, 4] / shrink
        det_ymax = image_shape[1] * detection[:, 5] / shrink

        det = np.column_stack(
            (det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
        return det

    def flip_test(self, image, shrink):
        img = image.transpose(Image.FLIP_LEFT_RIGHT)
        det_f = self.detect_face(img, shrink)
        det_t = np.zeros(det_f.shape)
        # image.size: [width, height]
        det_t[:, 0] = image.size[0] - det_f[:, 2]
        det_t[:, 1] = det_f[:, 1]
        det_t[:, 2] = image.size[0] - det_f[:, 0]
        det_t[:, 3] = det_f[:, 3]
        det_t[:, 4] = det_f[:, 4]
        return det_t

    def multi_scale_test(self, image, max_shrink):
        # Shrink detecting is only used to detect big faces
        st = 0.5 if max_shrink >= 0.75 else 0.5 * max_shrink
        det_s = self.detect_face(image, st)
        index = np.where(
            np.maximum(det_s[:, 2] - det_s[:, 0] + 1,
                       det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
        det_s = det_s[index, :]
        # Enlarge one times
        bt = min(2, max_shrink) if max_shrink > 1 else (st + max_shrink) / 2
        det_b = self.detect_face(image, bt)

        # Enlarge small image x times for small faces
        if max_shrink > 2:
            bt *= 2
            while bt < max_shrink:
                det_b = np.row_stack((det_b, self.detect_face(image, bt)))
                bt *= 2
            det_b = np.row_stack((det_b, self.detect_face(image, max_shrink)))

        # Enlarged images are only used to detect small faces.
        if bt > 1:
            index = np.where(
                np.minimum(det_b[:, 2] - det_b[:, 0] + 1,
                           det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
            det_b = det_b[index, :]
        # Shrinked images are only used to detect big faces.
        else:
            index = np.where(
                np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                           det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
            det_b = det_b[index, :]
        return det_s, det_b

    def multi_scale_test_pyramid(self, image, max_shrink):
        # Use image pyramids to detect faces
        det_b = self.detect_face(image, 0.25)
        index = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

        st = [0.75, 1.25, 1.5, 1.75]
        for i in range(len(st)):
            if st[i] <= max_shrink:
                det_temp = self.detect_face(image, st[i])
                # Enlarged images are only used to detect small faces.
                if st[i] > 1:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                                   det_temp[:, 3] - det_temp[:, 1] + 1) <
                        100)[0]
                    det_temp = det_temp[index, :]
                # Shrinked images are only used to detect big faces.
                else:
                    index = np.where(
                        np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                                   det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                    det_temp = det_temp[index, :]
                det_b = np.row_stack((det_b, det_temp))
        return det_b


def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    if det.shape[0] == 0:
        dets = np.array([[10, 10, 20, 20, 0.002]])
        det = np.empty(shape=[0, 5])
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # nms
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            if det.shape[0] == 0:
                try:
                    dets = np.row_stack((dets, det_accu))
                except:
                    dets = det_accu
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                      axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    # Only keep 0.3 or more
    keep_index = np.where(dets[:, 4] >= 0.3)[0]
    dets = dets[keep_index, :]
    return dets


def get_shrink(height, width):
    """
    Args:
        height (int): image height.
        width (int): image width.
    """
    # avoid out of memory
    max_shrink_v1 = (0x7fffffff / 577.0 / (height * width))**0.5
    max_shrink_v2 = ((678 * 1024 * 2.0 * 2.0) / (height * width))**0.5

    def get_round(x, loc):
        str_x = str(x)
        if '.' in str_x:
            str_before, str_after = str_x.split('.')
            len_after = len(str_after)
            if len_after >= 3:
                str_final = str_before + '.' + str_after[0:loc]
                return float(str_final)
            else:
                return x

    max_shrink = get_round(min(max_shrink_v1, max_shrink_v2), 2) - 0.3
    if max_shrink >= 1.5 and max_shrink < 2:
        max_shrink = max_shrink - 0.1
    elif max_shrink >= 2 and max_shrink < 3:
        max_shrink = max_shrink - 0.2
    elif max_shrink >= 3 and max_shrink < 4:
        max_shrink = max_shrink - 0.3
    elif max_shrink >= 4 and max_shrink < 5:
        max_shrink = max_shrink - 0.4
    elif max_shrink >= 5:
        max_shrink = max_shrink - 0.5

    shrink = max_shrink if max_shrink < 1 else 1
    return shrink, max_shrink


def save_widerface_bboxes_matlab(image_path, bboxes_scores, output_dir):
    """
    Save predicted results, including bbox and score into text file.
    Args:
        image_path (string): file name.
        bboxes_scores (np.array|list): the predicted bboxed and scores, layout
            is (xmin, ymin, xmax, ymax, score)
        output_dir (string): output directory.
    """
    image_name = image_path.split('/')[-1]
    image_class = image_path.split('/')[-2]

    odir = os.path.join(output_dir, image_class)
    if not os.path.exists(odir):
        os.makedirs(odir)

    ofname = os.path.join(odir, '%s.txt' % (image_name[:-4]))
    f = open(ofname, 'w')
    f.write('{:s}\n'.format(image_class + '/' + image_name))
    f.write('{:d}\n'.format(bboxes_scores.shape[0]))
    for box_score in bboxes_scores:
        xmin, ymin, xmax, ymax, score = box_score
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(xmin, ymin, (
            xmax - xmin + 1), (ymax - ymin + 1), score))
    f.close()
    logger.info("The predicted result is saved as {}".format(ofname))


def save_widerface_bboxes_py(imid2path, bboxes_scores, output_dir):
    """
    Save predicted results, including bbox and score into text file.
    Args:
        imid2path (string): file name dict.
        bboxes_scores (np.array|list): the predicted bboxed and scores, layout
            is (xmin, ymin, xmax, ymax, score)
        output_dir (string): output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    predict_file = os.path.join(output_dir, 'pred_res.txt')
    f = open(predict_file, 'w')
    for dets, image_path in zip(bboxes_scores, imid2path.values()):
        image_name = image_path.split('/')[-1]
        image_class = image_path.split('/')[-2]
        f.write('{:s}\n'.format(image_class + '/' + image_name))
        f.write('{:d}\n'.format(dets.shape[0]))
        for box_score in dets:
            xmin, ymin, xmax, ymax, score = box_score
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'
                    .format(xmin, ymin, xmax, ymax, score))
    logger.info("The predicted result is saved as {}".format(predict_file))
    return predict_file


def calculate_ap_py(pred_res_file, gt_file, pred_dir):
    """
    Calculate ap according to pred_res_file and gt_file
    Args:
    pred_res_file: predict result file path
    gt_file: annotation file path
    pred_dir: predict result output file directory
    """

    def calIOU(rect1, rect2):
        lt_x = max(rect1[0], rect2[0])
        lt_y = max(rect1[1], rect2[1])
        rb_x = min(rect1[2], rect2[2])
        rb_y = min(rect1[3], rect2[3])
        if (rb_x > lt_x) and (rb_y > lt_y):
            intersection = (rb_x - lt_x) * (rb_y - lt_y)
        else:
            return 0

        area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

        intersection = min(intersection, area1, area2)
        union = area1 + area2 - intersection
        return float(intersection) / union

    def isSameFace(face_gt, face_pred):
        iou = calIOU(face_gt, face_pred)
        return iou >= 0.5

    def evaluationSingleImage(faces_gt, faces_pred):
        pred_is_true = [False] * len(faces_pred)
        gt_been_pred = [False] * len(faces_gt)
        for i in range(len(faces_pred)):
            isFace = False
            for j in range(len(faces_gt)):
                if gt_been_pred[j] == 0:
                    isFace = isSameFace(faces_gt[j], faces_pred[i])
                    if isFace == 1:
                        gt_been_pred[j] = True
                        break
            pred_is_true[i] = isFace
        return pred_is_true

    # load ground truth files
    with open(gt_file, 'r') as f:
        gt_lines = f.readlines()
    faces_num_gt = 0
    pos_gt = 0
    faces_gt = {}
    while pos_gt < len(gt_lines):
        name_gt = gt_lines[pos_gt].strip('\n\t').split()[0]
        pos_gt += 1
        n_gt = int(gt_lines[pos_gt].strip('\n\t').split()[0])
        pos_gt += 1
        faces = []
        for i in range(0, n_gt):
            split_str = gt_lines[pos_gt].strip('\n\t').split(' ')
            x1_min = float(split_str[0])
            y1_min = float(split_str[1])
            w = float(split_str[2])
            h = float(split_str[3])
            faces.append([x1_min, y1_min, x1_min + w, y1_min + h])
            faces_num_gt += 1
            pos_gt += 1
        faces_gt[name_gt] = faces
    logger.info('The ground truth file load {} images'.format(len(faces_gt)))

    with open(pred_res_file, 'r') as f:
        pred_lines = f.readlines()
    pos_pred = 0
    score_res_pair = {}
    while pos_pred < len(pred_lines):
        # print pos_pred
        name_pred = pred_lines[pos_pred].strip('\n\t').split()[0]
        pos_pred += 1
        n_pred = int(pred_lines[pos_pred].strip('\n\t').split()[0])
        pos_pred += 1
        pred_faces = []
        for i in range(0, n_pred):
            line = pred_lines[pos_pred].strip('\n\t').split()
            face = []
            for j in range(len(line)):
                face.append(float(line[j]))
            pred_faces.append(face)
            pos_pred += 1

        gt_faces = faces_gt[name_pred]

        pred_is_true = evaluationSingleImage(gt_faces, pred_faces)

        for i in range(0, len(pred_is_true)):
            nowScore = pred_faces[i][-1]
            if score_res_pair.has_key(nowScore):
                score_res_pair[nowScore].append(int(pred_is_true[i]))
            else:
                score_res_pair[nowScore] = [int(pred_is_true[i])]
    keys = score_res_pair.keys()
    keys.sort(reverse=True)

    res_file = os.path.join(pred_dir, "pred_result.txt")
    outfile = open(res_file, 'w')
    tp_num = 0
    predict_num = 0
    precision_list = []
    recall_list = []
    outfile.write("recall falsePositiveNum precision scoreThreshold\n")
    for i in range(len(keys)):
        k = keys[i]
        v = score_res_pair[k]
        predict_num += len(v)
        tp_num += sum(v)
        fp_num = predict_num - tp_num
        recall = float(tp_num) / faces_num_gt
        precision = float(tp_num) / predict_num
        outfile.write('{} {} {} {}\n'.format(recall, fp_num, precision, k))
        precision_list.append(float(tp_num) / predict_num)
        recall_list.append(recall)
    ap = precision_list[0] * recall_list[0]
    for i in range(1, len(precision_list)):
        ap += precision_list[i] * (recall_list[i] - recall_list[i - 1])
    outfile.write('AP={}\n'.format(ap))

    logger.info(
        "AP = {}\nFor more details, please checkout the evaluation res at {},\n"
        .format(ap, res_file))
    outfile.close()


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    if use_default_label or anno_file is None \
            or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "wider-face categories.".format(anno_file))
        return vocall_category_info(with_background)
    else:
        logger.info("Load categories from {}".format(anno_file))
        return get_category_info_from_anno(anno_file, with_background)


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


def vocall_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of mixup voc dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    label_map = widerface_label(with_background)
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    if with_background:
        cats.insert(0, 'background')

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name
