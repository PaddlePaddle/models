from __future__ import print_function
import sys
import os
import math
import random
import functools
import io
import time
import codecs
import numpy as np
import paddle
import paddle.fluid as fluid
import cv2
import copy
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

from src.models.model_builder import ModelPhase
from src.utils.config import cfg
from .data_utils import GeneratorEnqueuer


class BaseSeg(object):
    def __init__(self, file_list, data_dir, shuffle=False, mode=ModelPhase.TRAIN, base_size=1024, crop_size=769, rand_scale=True):
        self.mode = mode
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.shuffle_seed = 0

        self.crop_size = crop_size  
        self.base_size = base_size  # short edge when training
        self.rand_scale = rand_scale

        # NOTE: Please ensure file list was save in UTF-8 coding format
        with codecs.open(file_list, 'r', 'utf-8') as flist:
            self.lines = [line.strip() for line in flist]
            self.all_lines = copy.deepcopy(self.lines)
            if shuffle and cfg.NUM_TRAINERS > 1:
                np.random.RandomState(self.shuffle_seed).shuffle(self.all_lines)
            elif shuffle:
                np.random.shuffle(self.lines)
        self.num_trainers= cfg.NUM_TRAINERS
        self.trainer_id=cfg.TRAINER_ID

    def generator(self):
        if self.shuffle and cfg.NUM_TRAINERS > 1:
            np.random.RandomState(self.shuffle_seed).shuffle(self.all_lines)
            num_lines = len(self.all_lines) // cfg.NUM_TRAINERS
            self.lines = self.all_lines[num_lines * cfg.TRAINER_ID: num_lines * (cfg.TRAINER_ID + 1)]
            self.shuffle_seed += 1
        elif self.shuffle:
            np.random.shuffle(self.lines)

        for line in self.lines:
            yield self.process_image(line, self.data_dir, self.mode)

    def sharding_generator(self, pid=0, num_processes=1):
        """
        Use line id as shard key for multiprocess io
        It's a normal generator if pid=0, num_processes=1
        """
        for index, line in enumerate(self.lines):
            # Use index and pid to shard file list
                if index % num_processes == pid:
                    yield self.process_image(line, self.data_dir, self.mode)

    def batch_reader(self, batch_size):
        br = self.batch(self.reader, batch_size)
        for batch in br:
            yield batch[0], batch[1], batch[2]

    def multiprocess_generator(self, max_queue_size=32, num_processes=8):
        # Re-shuffle file list
        if self.shuffle and cfg.NUM_TRAINERS > 1:
            np.random.RandomState(self.shuffle_seed).shuffle(self.all_lines)
            num_lines = len(self.all_lines) // self.num_trainers
            self.lines = self.all_lines[num_lines * self.trainer_id: num_lines * (self.trainer_id + 1)]
            self.shuffle_seed += 1
        elif self.shuffle:
            np.random.shuffle(self.lines)

        # Create multiple sharding generators according to num_processes for multiple processes
        generators = []
        for pid in range(num_processes):
            generators.append(self.sharding_generator(pid, num_processes))

        try:
            enqueuer = GeneratorEnqueuer(generators)
            enqueuer.start(max_queue_size=max_queue_size, workers=num_processes)
            while True:
                generator_out = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        generator_out = enqueuer.queue.get(timeout=5)
                        break
                    else:
                        time.sleep(0.01)
                if generator_out is None:
                    break
                yield generator_out
        finally:
            if enqueuer is not None:
                enqueuer.stop()

    def batch(self, reader, batch_size, is_test=False, drop_last=False):
        def batch_reader(is_test=False, drop_last=drop_last):
            if is_test:
                imgs, grts, img_names, valid_shapes, org_shapes = [], [], [], [], []
                for img, grt, img_name, valid_shape, org_shape in reader():
                    imgs.append(img)
                    grts.append(grt)
                    img_names.append(img_name)
                    valid_shapes.append(valid_shape)
                    org_shapes.append(org_shape)
                    if len(imgs) == batch_size:
                        yield np.array(imgs), np.array(
                            grts), img_names, np.array(valid_shapes), np.array(
                                org_shapes)
                        imgs, grts, img_names, valid_shapes, org_shapes = [], [], [], [], []

                if not drop_last and len(imgs) > 0:
                    yield np.array(imgs), np.array(grts), img_names, np.array(
                        valid_shapes), np.array(org_shapes)
            else:
                imgs, labs, ignore = [], [], []
                bs = 0
                for img, lab, ig in reader():
                    imgs.append(img)
                    labs.append(lab)
                    ignore.append(ig)
                    bs += 1
                    if bs == batch_size:
                        yield np.array(imgs), np.array(labs), np.array(ignore)
                        bs = 0
                        imgs, labs, ignore = [], [], []

                if not drop_last and bs > 0:
                    yield np.array(imgs), np.array(labs), np.array(ignore)

        return batch_reader(is_test, drop_last)

    def load_image(self, line, src_dir, mode=ModelPhase.TRAIN):
        raise NotImplemented

    def pil_imread(self, file_path):
        """read pseudo-color label"""
        im = Image.open(file_path)
        return np.asarray(im)

    def cv2_imread(self, file_path, flag=cv2.IMREAD_COLOR):
        # resolve cv2.imread open Chinese file path issues on Windows Platform.
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)

    def normalize_image(self, img):
        img = img.transpose((2, 0, 1)).astype('float32') / 255.0
        img_mean = np.array(cfg.MEAN).reshape((len(cfg.MEAN), 1, 1))
        img_std = np.array(cfg.STD).reshape((len(cfg.STD), 1, 1))
        img -= img_mean
        img /= img_std

        return img

    def process_image(self, line, data_dir, mode):
        """ process_image """
        img, grt, img_name, grt_name = self.load_image( line, data_dir, mode=mode)  # img.type: numpy.array, grt.type: numpy.array
        if mode == ModelPhase.TRAIN:
            # numpy.array convert to  PIL.Image 
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            grt = Image.fromarray(grt.astype('uint8')).convert('L')
            
            crop_size = self.crop_size
            # random scale 
            if self.rand_scale:
                short_size = random.randint(int(self.base_size * cfg.DATAAUG.RAND_SCALE_MIN), int(self.base_size * cfg.DATAAUG.RAND_SCALE_MAX))
            else:
                short_size = self.base_size
            w, h = img.size
            if h > w:
                out_w = short_size
                out_h = int(1.0 * h / w * out_w)
            else:
                out_h = short_size
                out_w = int(1.0 * w / h * out_h)
            img = img.resize((out_w, out_h), Image.BILINEAR)
            grt = grt.resize((out_w, out_h), Image.NEAREST)

            # rand flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                grt = grt.transpose(Image.FLIP_LEFT_RIGHT)

            # padding
            if short_size < crop_size:
                pad_h = crop_size - out_h if out_h < crop_size else 0
                pad_w = crop_size - out_w if out_w < crop_size else 0
                img = ImageOps.expand(img, border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), fill=0)
                grt = ImageOps.expand(grt, border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), fill=cfg.DATASET.IGNORE_INDEX)

            # random crop
            w, h = img.size
            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)
            img = img.crop((x, y, x + crop_size, y + crop_size))
            grt = grt.crop((x, y, x + crop_size, y + crop_size))


            # gaussian blur
            if cfg.DATAAUG_EXTRA:
                if random.random() > 0.7:
                    img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

            # PIL.Image -> cv2
            img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            grt = np.array(grt)
            
        elif ModelPhase.is_eval(mode):
            org_shape = [img.shape[0], img.shape[1]]  # 1024 x 2048 for cityscapes

        elif ModelPhase.is_visual(mode):
            org_shape = [img.shape[0], img.shape[1]]
            #img, grt = resize(img, grt, mode=mode)
            valid_shape = [img.shape[0], img.shape[1]]
            #img, grt = rand_crop(img, grt, mode=mode)
        else:
            raise ValueError("Dataset mode={} Error!".format(mode))

        # Normalize image
        img = self.normalize_image(img)

        if ModelPhase.is_train(mode) or ModelPhase.is_eval(mode):
            grt = np.expand_dims(np.array(grt).astype('int32'), axis=0)
            ignore = (grt != cfg.DATASET.IGNORE_INDEX).astype('int32')


        if ModelPhase.is_train(mode):
            return (img, grt, ignore)
        elif ModelPhase.is_eval(mode):
            return (img, grt, ignore)
        elif ModelPhase.is_visual(mode):
            return (img, grt, img_name, valid_shape, org_shape)
