from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# GPU memory garbage collection optimization flags
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
import sys
import time
import argparse
import functools
import pprint
import cv2
import numpy as np
import paddle
import paddle.fluid as fluid
import math

from src.utils.config import cfg
from src.utils.timer import Timer, calculate_eta
from src.models.model_builder import build_model
from src.models.model_builder import ModelPhase
from src.datasets import build_dataset
from src.utils.metrics import ConfusionMatrix


def parse_args():
    parser = argparse.ArgumentParser(description='SemsegPaddle')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str)
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        help='Use gpu or cpu',
        action='store_true',
        default=False)
    parser.add_argument(
        '--use_mpio',
        dest='use_mpio',
        help='Use multiprocess IO or not',
        action='store_true',
        default=False)
    parser.add_argument(
        'opts',
        help='See utils/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
            '--multi_scales',
            dest='multi_scales',
            help='Use multi_scales for eval',
            action='store_true',
            default=False)
    parser.add_argument(
            '--flip',
            dest='flip',
            help='flip the image or not',
            action='store_true',
            default=False)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def evaluate(cfg, ckpt_dir=None, use_gpu=False, use_mpio=False, multi_scales=False, flip=False,  **kwargs):
    np.set_printoptions(precision=5, suppress=True)
    
    num_classes = cfg.DATASET.NUM_CLASSES
    base_size = cfg.TEST.BASE_SIZE
    crop_size = cfg.TEST.CROP_SIZE
    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    dataset = build_dataset(cfg.DATASET.DATASET_NAME,
        file_list=cfg.DATASET.VAL_FILE_LIST,
        mode=ModelPhase.EVAL,
        data_dir=cfg.DATASET.DATA_DIR)

    def data_generator():
        #TODO: check is batch reader compatitable with Windows
        if use_mpio:
            data_gen = dataset.multiprocess_generator(
                num_processes=cfg.DATALOADER.NUM_WORKERS,
                max_queue_size=cfg.DATALOADER.BUF_SIZE)
        else:
            data_gen = dataset.generator()

        for b in data_gen:
            yield b[0], b[1], b[2]

    py_reader, avg_loss, out, grts, masks = build_model(
        test_prog, startup_prog, phase=ModelPhase.EVAL)

    py_reader.decorate_sample_generator(
        data_generator, drop_last=False, batch_size=cfg.EVAL_BATCH_SIZE, places=fluid.cuda_places())

    # Get device environment
    places = fluid.cuda_places() if use_gpu else fluid.cpu_places()
    place = places[0]
    dev_count = len(places)
    print("#Device count: {}".format(dev_count))

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    test_prog = test_prog.clone(for_test=True)

    ckpt_dir = cfg.TEST.TEST_MODEL if not ckpt_dir else ckpt_dir

    if ckpt_dir is not None:
        filename= '{}_{}_{}_epoch_{}.pdparams'.format(str(cfg.MODEL.MODEL_NAME),
                                                  str(cfg.MODEL.BACKBONE), str(cfg.DATASET.DATASET_NAME), cfg.SOLVER.NUM_EPOCHS)
        print("loading testing model file: {}/{}".format(ckpt_dir, filename))
        fluid.io.load_params(exe, ckpt_dir, main_program=test_prog, filename=filename)

    # Use streaming confusion matrix to calculate mean_iou
    np.set_printoptions(
        precision=4, suppress=True, linewidth=160, floatmode="fixed")
    conf_mat = ConfusionMatrix(cfg.DATASET.NUM_CLASSES, streaming=True)
    
    #fetch_list: return of the model
    fetch_list = [avg_loss.name, out.name]
    num_images = 0
    step = 0
    all_step = cfg.DATASET.VAL_TOTAL_IMAGES // cfg.EVAL_BATCH_SIZE 
    timer = Timer()
    timer.start()
    for data in py_reader():
        mask = np.array(data[0]['mask'])
        label = np.array(data[0]['label'])
        image_org = np.array(data[0]['image'])
        image = np.transpose(image_org, (0, 2, 3, 1)) # BCHW->BHWC
        image = np.squeeze(image)

        if cfg.TEST.SLIDE_WINDOW:
            if not multi_scales:
                scales = [1.0]
            else:
                scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if cfg.DATASET.DATASET_NAME == 'cityscapes' else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
                #scales = [0.75, 1.0, 1.25] # fast multi-scale testing
        
            #strides
            stride = int(crop_size *1.0 / 3)  # 1/3 > 2/3 > 1/2 for input_size: 769 x 769
            h, w = image.shape[0:2]
            scores = np.zeros(shape=[num_classes, h, w], dtype='float32')

            for scale in scales:
                long_size = int(math.ceil(base_size * scale))
                if h > w:
                    height = long_size
                    width = int(1.0 * w * long_size / h + 0.5)
                    short_size = width
                else:
                    width = long_size
                    height = int(1.0 * h * long_size / w + 0.5)
                    short_size = height
                # print('org_img_size: {}x{}, rescale_img_size: {}x{}'.format(h, w, height, width))
                cur_img = image_resize(image, height, width)
                # pading
                if long_size <= crop_size:
                    pad_img = pad_single_image(cur_img, crop_size)
                    label_feed, mask_feed = get_feed(pad_img)
                    pad_img = mapper_image(pad_img)
                    loss, pred1 = exe.run(
                            test_prog, 
                            feed={'image':pad_img, 'label':label_feed, 'mask':mask_feed}, 
                            fetch_list = fetch_list,
                            return_numpy=True)
                    pred1 = np.array(pred1)
                    outputs = pred1[:, :, :height, :width]
                    if flip:
                        pad_img_flip = flip_left_right_image(cur_img)
                        pad_img_flip = pad_single_image(pad_img_flip, crop_size)
                        label_feed, mask_feed = get_feed(pad_img_flip)

                        pad_img_flip = mapper_image(pad_img_flip)
                        loss, pred1 = exe.run(
                                test_prog,
                                feed={'image':pad_img_flip, 'label':label_feed, 'mask':mask_feed},
                                fetch_list = fetch_list,
                                return_numpy=True)
                        pred1 = np.flip(pred1, 3)
                        outputs += pred1[:, :, :height, :width]
                else:
                    if short_size < crop_size:
                        pad_img = pad_single_image(cur_img, crop_size)
                    else:
                        pad_img = cur_img
                    ph, pw = pad_img.shape[0:2]

                    #slid window
                    h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                    w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                    outputs = np.zeros(shape=[1, num_classes, ph, pw], dtype='float32')
                    count_norm = np.zeros(shape=[1, 1, ph, pw], dtype='int32')
                    for idh in range(h_grids):
                        for idw in range(w_grids):
                            h0 = idh * stride
                            w0 = idw * stride
                            h1 = min(h0 + crop_size, ph)
                            w1 = min(w0 + crop_size, pw)
                            #print('(h0,w0,h1,w1):({},{},{},{})'.format(h0, w0, h1, w1))
                            crop_img = crop_image(pad_img, h0, w0, h1, w1)
                            pad_crop_img = pad_single_image(crop_img, crop_size)
                            label_feed, mask_feed = get_feed(pad_crop_img)
                            pad_crop_img = mapper_image(pad_crop_img)
                            loss, pred1 = exe.run(
                                    test_prog, 
                                    feed={'image':pad_crop_img, 'label':label_feed, 'mask':mask_feed},
                                    fetch_list = fetch_list,
                                    return_numpy=True)
                            pred1 = np.array(pred1)
                            outputs[:, :, h0:h1, w0:w1] += pred1[:, :, 0:h1-h0, 0:w1-w0]
                            count_norm[:, :, h0:h1, w0:w1] += 1
                            if flip:
                                pad_img_flip = flip_left_right_image(crop_img)
                                pad_img_flip = pad_single_image(pad_img_flip, crop_size)
                                label_feed, mask_feed = get_feed(pad_img_flip)
                                pad_img_flip = mapper_image(pad_img_flip)
                                loss, pred1 = exe.run(
                                        test_prog,
                                        feed={'image':pad_img_flip, 'label':label_feed, 'mask':mask_feed},
                                        fetch_list = fetch_list,
                                        return_numpy = True)
                                pred1 = np.flip(pred1, 3)
                                outputs[:, :, h0:h1, w0:w1] += pred1[:, :, 0:h1-h0, 0:w1-w0]
                                count_norm[:, :, h0:h1, w0:w1] += 1
                    
                    outputs = 1.0 * outputs / count_norm
                    outputs = outputs[:, :, :height, :width]
                with fluid.dygraph.guard():
                    outputs = fluid.dygraph.to_variable(outputs)
                    outputs = fluid.layers.resize_bilinear(outputs, out_shape=[h, w])
                    score = outputs.numpy()[0]
                    scores += score
        else: 
            # taking the original image as the model input     
            loss, pred = exe.run(
                    test_prog,
                    feed={'image':image_org, 'label':label, 'mask':mask},
                    fetch_list = fetch_list,
                    return_numpy = True)
            scores = pred[0]
        # computing IoU with all scale result
        pred = np.argmax(scores, axis=0).astype('int64')
        pred = pred[np.newaxis, :, :, np.newaxis]
        step += 1
        num_images += pred.shape[0]
        conf_mat.calculate(pred, label, mask)
        _, iou = conf_mat.mean_iou()
        _, acc = conf_mat.accuracy()

        print("[EVAL] step={}/{} acc={:.4f} IoU={:.4f}".format(step, all_step, acc, iou))

    category_iou, avg_iou = conf_mat.mean_iou()
    category_acc, avg_acc = conf_mat.accuracy()
    print("[EVAL] #image={} acc={:.4f} IoU={:.4f}".format(num_images, avg_acc, avg_iou))
    print("[EVAL] Category IoU:", category_iou)
    print("[EVAL] Category Acc:", category_acc)
    print("[EVAL] Kappa:{:.4f}".format(conf_mat.kappa()))
    print("flip = ", flip)
    print("scales = ", scales)

    return category_iou, avg_iou, category_acc, avg_acc

def image_resize(image, height, width):
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image

def pad_single_image(image, crop_size):
    h, w  = image.shape[0:2]
    pad_h = crop_size - h if h < crop_size else 0
    pad_w = crop_size - w if w < crop_size else 0
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,value=0)
    return image

def mapper_image(image):
    # HxWx3 -> 3xHxW -> 1x3xHxW
    image_array = np.transpose(image, (2, 0, 1))
    image_array = image_array.astype('float32')
    image_array = image_array[np.newaxis, :]
    return image_array

def flip_left_right_image(image):
    return cv2.flip(image, 1)

def get_feed(image):
    h, w = image.shape[0:2]
    return np.zeros([1, 1, h, w], dtype='int32'), np.zeros([1, 1, h, w], dtype='int32')

def crop_image(image, h0, w0, h1, w1):
    return image[h0:h1, w0:w1, :]

def main():
    args = parse_args()
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    cfg.check_and_infer()
    print(pprint.pformat(cfg))
    evaluate(cfg, **args.__dict__)


if __name__ == '__main__':
    main()
