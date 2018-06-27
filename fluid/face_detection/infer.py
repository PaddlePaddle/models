import os
import time
import numpy as np
import argparse
import functools
from PIL import Image
from PIL import ImageDraw

import paddle
import paddle.fluid as fluid
import reader
from pyramidbox import PyramidBox
from utility import add_arguments, print_arguments
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',          bool,  True,     "Whether use GPU.")
add_arg('use_pyramidbox',   bool,  True,     "Whether use PyramidBox model.")
add_arg('confs_threshold',  float, 0.25,     "Confidence threshold to draw bbox.")
add_arg('image_path',       str,   '',       "The data root path.")
add_arg('model_dir',        str,   '',        "The model path.")
add_arg('slice_num',        int,   1,         "Split number.")
add_arg('slice_index',     int,    1,         "Index.")
# yapf: enable


def draw_bounding_box_on_image(image_path, nms_out, confs_threshold):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for dt in nms_out:
        xmin, ymin, xmax, ymax, score = dt
        if score < confs_threshold:
            continue
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=4,
            fill='red')
    image_name = image_path.split('/')[-1]
    image_class = image_path.split('/')[-2]
    print("image with bbox drawed saved as {}".format(image_name))
    image.save('./infer_results/' + image_class.encode('utf-8') + '/' +
               image_name.encode('utf-8'))


def write_to_txt(image_path, f, nms_out):
    image_name = image_path.split('/')[-1]
    image_class = image_path.split('/')[-2]
    f.write('{:s}\n'.format(
        image_class.encode('utf-8') + '/' + image_name.encode('utf-8')))
    f.write('{:d}\n'.format(nms_out.shape[0]))
    for dt in nms_out:
        xmin, ymin, xmax, ymax, score = dt
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(xmin, ymin, (
            xmax - xmin + 1), (ymax - ymin + 1), score))
    print("image infer result saved {}".format(image_name[:-4]))


def get_round(x, loc):
    str_x = str(x)
    if '.' in str_x:
        len_after = len(str_x.split('.')[1])
        str_before = str_x.split('.')[0]
        str_after = str_x.split('.')[1]
        if len_after >= 3:
            str_final = str_before + '.' + str_after[0:loc]
            return float(str_final)
        else:
            return x


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

        # get needed merge det and delete these det
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


def image_preprocess(image):
    img = np.array(image)
    # HWC to CHW
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    # RBG to BGR
    img = img[[2, 1, 0], :, :]
    img = img.astype('float32')
    img -= np.array(
        [104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
    img = img * 0.007843
    img = [img]
    img = np.array(img)
    return img


def detect_face(image, shrink):
    image_shape = [3, image.size[1], image.size[0]]
    num_classes = 2
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    if shrink != 1:
        image = image.resize((int(image_shape[2] * shrink),
                              int(image_shape[1] * shrink)), Image.ANTIALIAS)
        image_shape = [
            image_shape[0], int(image_shape[1] * shrink),
            int(image_shape[2] * shrink)
        ]
    print "image_shape:", image_shape
    img = image_preprocess(image)

    scope = fluid.core.Scope()
    main_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.scope_guard(scope):
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program, startup_program):
                fetches = []
                network = PyramidBox(
                    image_shape,
                    num_classes,
                    sub_network=args.use_pyramidbox,
                    is_infer=True)
                infer_program, nmsed_out = network.infer(main_program)
                fetches = [nmsed_out]
                fluid.io.load_persistables(
                    exe, args.model_dir, main_program=main_program)

                detection, = exe.run(infer_program,
                                     feed={'image': img},
                                     fetch_list=fetches,
                                     return_numpy=False)
                detection = np.array(detection)
    # layout: xmin, ymin, xmax. ymax, score
    if detection.shape == (1, ):
        print "no face detected"
        return np.array([[0, 0, 0, 0, 0]])
    det_conf = detection[:, 1]
    det_xmin = image_shape[2] * detection[:, 2] / shrink
    det_ymin = image_shape[1] * detection[:, 3] / shrink
    det_xmax = image_shape[2] * detection[:, 4] / shrink
    det_ymax = image_shape[1] * detection[:, 5] / shrink

    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


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
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_shrink >= 0.75 else 0.5 * max_shrink
    det_s = detect_face(image, st)
    index = np.where(
        np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1)
        > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_shrink) if max_shrink > 1 else (st + max_shrink) / 2
    det_b = detect_face(image, bt)

    # enlarge small image x times for small face
    if max_shrink > 2:
        bt *= 2
        while bt < max_shrink:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(image, max_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(
            np.minimum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    return det_s, det_b


def multi_scale_test_pyramid(image, max_shrink):
    # shrink detecting and shrink only detect big face
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [0.5, 0.75, 1.25, 1.5, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


def get_im_shrink(image_shape):
    max_shrink_v1 = (0x7fffffff / 577.0 /
                     (image_shape[1] * image_shape[2]))**0.5
    max_shrink_v2 = (
        (678 * 1024 * 2.0 * 2.0) / (image_shape[1] * image_shape[2]))**0.5
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

    print 'max_shrink = ', max_shrink
    shrink = max_shrink if max_shrink < 1 else 1
    print "shrink = ", shrink

    return shrink, max_shrink


def infer(args, batch_size, data_args):
    if not os.path.exists(args.model_dir):
        raise ValueError("The model path [%s] does not exist." %
                         (args.model_dir))

    infer_reader = paddle.batch(
        reader.test(data_args, file_list), batch_size=batch_size)

    for batch_id, img in enumerate(infer_reader()):
        image = img[0][0]
        image_path = img[0][1]

        # image.size: [width, height]
        image_shape = [3, image.size[1], image.size[0]]

        shrink, max_shrink = get_im_shrink(image_shape)

        det0 = detect_face(image, shrink)
        det1 = flip_test(image, shrink)
        [det2, det3] = multi_scale_test(image, max_shrink)
        det4 = multi_scale_test_pyramid(image, max_shrink)
        det = np.row_stack((det0, det1, det2, det3, det4))
        dets = bbox_vote(det)

        image_name = image_path.split('/')[-1]
        image_class = image_path.split('/')[-2]
        if not os.path.exists('./infer_results/' + image_class.encode('utf-8')):
            os.makedirs('./infer_results/' + image_class.encode('utf-8'))

        f = open('./infer_results/' + image_class.encode('utf-8') + '/' +
                 image_name.encode('utf-8')[:-4] + '.txt', 'w')
        write_to_txt(image_path, f, dets)
        # draw_bounding_box_on_image(image_path, dets, args.confs_threshold)
    print "Done"


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/WIDERFACE/WIDER_val/images/'
    file_list = 'label/val_gt_widerface.res'

    data_args = reader.Settings(
        data_dir=data_dir,
        mean_value=[104., 117., 123],
        apply_distort=False,
        apply_expand=False,
        slice_num=args.slice_num,
        slice_index=args.slice_index)
    infer(args, batch_size=1, data_args=data_args)
