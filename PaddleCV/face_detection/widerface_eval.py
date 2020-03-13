#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import argparse
import functools
from PIL import Image


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

import paddle.fluid as fluid
import reader
from pyramidbox import PyramidBox
from visualize import draw_bboxes
from utility import add_arguments, print_arguments
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('use_gpu',         bool,  True,                              "Whether use GPU or not.")
add_arg('use_pyramidbox',  bool,  True,                              "Whether use PyramidBox model.")
add_arg('data_dir',        str,   'data/WIDER_val/images/',          "The validation dataset path.")
add_arg('model_dir',       str,   '',                                "The model path.")
add_arg('pred_dir',        str,   'pred',                            "The path to save the evaluation results.")
add_arg('file_list',       str,   'data/wider_face_split/wider_face_val_bbx_gt.txt', "The validation dataset path.")
add_arg('infer',           bool,  False,                             "Whether do infer or eval.")
add_arg('confs_threshold', float, 0.15,                              "Confidence threshold to draw bbox.")
add_arg('image_path',      str,   '',                                "The image used to inference and visualize.")
# yapf: enable


def infer(args, config):
    model_dir = args.model_dir
    pred_dir = args.pred_dir
    if not os.path.exists(model_dir):
        raise ValueError("The model path [%s] does not exist." % (model_dir))

    if args.infer:
        image_path = args.image_path
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        shrink, max_shrink = get_shrink(image.size[1], image.size[0])

        det0 = detect_face(image, shrink)
        if args.use_gpu:
            det1 = flip_test(image, shrink)
            [det2, det3] = multi_scale_test(image, max_shrink)
            det4 = multi_scale_test_pyramid(image, max_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)
        else:
            # when infer on cpu, use a simple case
            dets = det0

        keep_index = np.where(dets[:, 4] >= args.confs_threshold)[0]
        dets = dets[keep_index, :]
        draw_bboxes(image_path, dets[:, 0:4])
    else:
        test_reader = reader.test(config, args.file_list)
        for image, image_path in test_reader():
            shrink, max_shrink = get_shrink(image.size[1], image.size[0])

            det0 = detect_face(image, shrink)
            det1 = flip_test(image, shrink)
            [det2, det3] = multi_scale_test(image, max_shrink)
            det4 = multi_scale_test_pyramid(image, max_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)

            save_widerface_bboxes(image_path, dets, pred_dir)

        print("Finish evaluation.")


def save_widerface_bboxes(image_path, bboxes_scores, output_dir):
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
    print("The predicted result is saved as {}".format(ofname))


def detect_face(image, shrink):
    image_shape = [3, image.size[1], image.size[0]]
    if shrink != 1:
        h, w = int(image_shape[1] * shrink), int(image_shape[2] * shrink)
        image = image.resize((w, h), Image.ANTIALIAS)
        image_shape = [3, h, w]

    img = np.array(image)
    img = reader.to_chw_bgr(img)
    mean = [104., 117., 123.]
    scale = 0.007843
    img = img.astype('float32')
    img -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    img = img * scale
    img = [img]
    img = np.array(img)

    detection, = exe.run(infer_program,
                         feed={'image': img},
                         fetch_list=fetches,
                         return_numpy=False)
    detection = np.array(detection)
    # layout: xmin, ymin, xmax. ymax, score
    if np.prod(detection.shape) == 1:
        print("No face detected")
        return np.array([[0, 0, 0, 0, 0]])
    det_conf = detection[:, 1]
    det_xmin = image_shape[2] * detection[:, 2] / shrink
    det_ymin = image_shape[1] * detection[:, 3] / shrink
    det_xmax = image_shape[2] * detection[:, 4] / shrink
    det_ymax = image_shape[1] * detection[:, 5] / shrink

    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
    return det


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
    return dets


def flip_test(image, shrink):
    img = image.transpose(Image.FLIP_LEFT_RIGHT)
    det_f = detect_face(img, shrink)
    det_t = np.zeros(det_f.shape)
    # image.size: [width, height]
    det_t[:, 0] = image.size[0] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.size[0] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(image, max_shrink):
    # Shrink detecting is only used to detect big faces
    st = 0.5 if max_shrink >= 0.75 else 0.5 * max_shrink
    det_s = detect_face(image, st)
    index = np.where(
        np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1)
        > 30)[0]
    det_s = det_s[index, :]
    # Enlarge one times
    bt = min(2, max_shrink) if max_shrink > 1 else (st + max_shrink) / 2
    det_b = detect_face(image, bt)

    # Enlarge small image x times for small faces
    if max_shrink > 2:
        bt *= 2
        while bt < max_shrink:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(image, max_shrink)))

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


def multi_scale_test_pyramid(image, max_shrink):
    # Use image pyramids to detect faces
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [0.75, 1.25, 1.5, 1.75]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # Enlarged images are only used to detect small faces.
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            # Shrinked images are only used to detect big faces.
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


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
    elif max_shrink <= 0.1:
        max_shrink = 0.1

    shrink = max_shrink if max_shrink < 1 else 1
    return shrink, max_shrink


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    config = reader.Settings(data_dir=args.data_dir)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    main_program = fluid.Program()
    startup_program = fluid.Program()
    image_shape = [3, 1024, 1024]
    with fluid.program_guard(main_program, startup_program):
        network = PyramidBox(
            data_shape=image_shape,
            sub_network=args.use_pyramidbox,
            is_infer=True)
        infer_program, nmsed_out = network.infer(main_program)
        fetches = [nmsed_out]
        exe.run(startup_program)
        fluid.io.load_persistables(
            exe, args.model_dir, main_program=infer_program)
        # save model and program
        #fluid.io.save_inference_model('pyramidbox_model',
        #    ['image'], [nmsed_out], exe, main_program=infer_program,
        #    model_filename='model', params_filename='params')
    infer(args, config)
