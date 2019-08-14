#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import random
import cv2
import sys
import numpy as np
import gc
import copy
import multiprocessing

import logging
logger = logging.getLogger(__name__)

try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO

from .reader_utils import DataReader
from models.ctcn.ctcn_utils import box_clamp1D, box_iou1D, BoxCoder

python_ver = sys.version_info

#random.seed(0)
#np.random.seed(0)


class CTCNReader(DataReader):
    """
    Data reader for C-TCN model, which was stored as features extracted by prior networks
    dataset cfg: img_size, the temporal dimension size of input data
                 root, the root dir of data
                 snippet_length, snippet length when sampling
                 filelist, the file list storing id and annotations of each data item
                 rgb, the dir of rgb data
                 flow, the dir of optical flow data
                 batch_size, batch size of input data
                 num_threads, number of threads of data processing

    """

    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        self.img_size = cfg.MODEL.img_size  # 512
        self.snippet_length = cfg.MODEL.snippet_length  # 1
        self.root = cfg.MODEL.root  # root dir of data
        self.filelist = cfg[mode.upper()]['filelist']
        self.rgb = cfg[mode.upper()]['rgb']
        self.flow = cfg[mode.upper()]['flow']
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.num_threads = cfg[mode.upper()]['num_threads']
        if (mode == 'test') or (mode == 'infer'):
            self.num_threads = 1  # set num_threads as 1 for test and infer

    def random_move(self, img, o_boxes, labels):
        boxes = np.array(o_boxes)
        mask = np.zeros(img.shape[0])
        for i in boxes:
            for j in range(i[0].astype('int'),
                           min(i[1].astype('int'), img.shape[0])):
                mask[j] = 1
        mask = (mask == 0)
        bg = img[mask]
        bg_len = bg.shape[0]
        if bg_len < 5:
            return img, boxes, labels
        insert_place = random.sample(range(bg_len), len(boxes))
        index = np.argsort(insert_place)
        new_img = bg[0:insert_place[index[0]], :]
        new_boxes = []
        new_labels = []

        for i in range(boxes.shape[0]):
            new_boxes.append([
                new_img.shape[0],
                new_img.shape[0] + boxes[index[i]][1] - boxes[index[i]][0]
            ])
            new_labels.append(labels[index[i]])
            new_img = np.concatenate(
                (new_img,
                 img[int(boxes[index[i]][0]):int(boxes[index[i]][1]), :]))
            if i < boxes.shape[0] - 1:
                new_img = np.concatenate(
                    (new_img,
                     bg[insert_place[index[i]]:insert_place[index[i + 1]], :]))
        new_img = np.concatenate(
            (new_img, bg[insert_place[index[len(boxes) - 1]]:, :]))
        del img, boxes, mask, bg, labels
        gc.collect()
        return new_img, new_boxes, new_labels

    def random_crop(self, img, boxes, labels, min_scale=0.3):
        boxes = np.array(boxes)
        labels = np.array(labels)
        imh, imw = img.shape[:2]
        params = [(0, imh)]
        for min_iou in (0, 0.1, 0.3, 0.5, 0.7, 0.9):
            for _ in range(100):
                scale = random.uniform(0.3, 1)
                h = int(imh * scale)

                y = random.randrange(imh - h)
                roi = [[y, y + h]]
                ious = box_iou1D(boxes, roi)
                if ious.min() >= min_iou:
                    params.append((y, h))
                    break
        y, h = random.choice(params)
        img = img[y:y + h, :]
        center = (boxes[:, 0] + boxes[:, 1]) / 2
        mask = (center[:] >= y) & (center[:] <= y + h)
        if mask.any():
            boxes = boxes[np.squeeze(mask.nonzero())] - np.array([[y, y]])
            boxes = box_clamp1D(boxes, 0, h)
            labels = labels[mask]
        else:
            boxes = [[0, 0]]
            labels = [0]
        return img, boxes, labels

    def resize(self, img, boxes, size, random_interpolation=False):
        '''Resize the input PIL image to given size.

        If boxes is not None, resize boxes accordingly.

        Args:
          img: image to be resized.
          boxes: (tensor) object boxes, sized [#obj,2].
          size: (tuple or int)
            - if is tuple, resize image to the size.
            - if is int, resize the shorter side to the size while maintaining the aspect ratio.
          random_interpolation: (bool) randomly choose a resize interpolation method.

        Returns:
          img: (cv2's numpy.ndarray) resized image.
          boxes: (tensor) resized boxes.

        Example:
        >> img, boxes = resize(img, boxes, 600)  # resize shorter side to 600
        '''
        h, w = img.shape[:2]
        if h == size:
            return img, boxes
        if h == 0:
            img = np.zeros((512, 402), np.float32)
            return img, boxes

        ow = w
        oh = size
        sw = 1
        sh = float(oh) / h
        method = random.choice([
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA
        ]) if random_interpolation else cv2.INTER_NEAREST
        img = cv2.resize(img, (ow, oh), interpolation=method)
        if boxes is not None:
            boxes = boxes * np.array([sh, sh])
        return img, boxes

    def transform(self, feats, boxes, labels, mode):
        feats = np.array(feats)
        boxes = np.array(boxes)
        labels = np.array(labels)
        #print('name {}, labels {}'.format(fname, labels))

        if mode == 'train':
            feats, boxes, labels = self.random_move(feats, boxes, labels)
            feats, boxes, labels = self.random_crop(feats, boxes, labels)
            feats, boxes = self.resize(
                feats, boxes, size=self.img_size, random_interpolation=True)
            h, w = feats.shape[:2]
            img = feats.reshape(1, h, w)
            Coder = BoxCoder()
            boxes, labels = Coder.encode(boxes, labels)
        if mode == 'test' or mode == 'valid':
            feats, boxes = self.resize(feats, boxes, size=self.img_size)
            h, w = feats.shape[:2]
            img = feats.reshape(1, h, w)
            Coder = BoxCoder()
            boxes, labels = Coder.encode(boxes, labels)
        return img, boxes, labels

    def load_file(self, fname):
        if python_ver < (3, 0):
            rgb_pkl = pickle.load(
                open(os.path.join(self.root, self.rgb, fname + '.pkl'), 'rb'))
            flow_pkl = pickle.load(
                open(os.path.join(self.root, self.flow, fname + '.pkl'), 'rb'))
        else:
            rgb_pkl = pickle.load(
                open(os.path.join(self.root, self.rgb, fname + '.pkl'), 'rb'),
                encoding='bytes')
            flow_pkl = pickle.load(
                open(os.path.join(self.root, self.flow, fname + '.pkl'), 'rb'),
                encoding='bytes')
        data_flow = np.array(flow_pkl[b'scores'])
        data_rgb = np.array(rgb_pkl[b'scores'])
        if data_flow.shape[0] < data_rgb.shape[0]:
            data_rgb = data_rgb[0:data_flow.shape[0], :]
        elif data_flow.shape[0] > data_rgb.shape[0]:
            data_flow = data_flow[0:data_rgb.shape[0], :]

        feats = np.concatenate((data_rgb, data_flow), axis=1)
        if feats.shape[0] == 0 or feats.shape[1] == 0:
            feats = np.zeros((512, 1024), np.float32)
            logger.info('### file loading len = 0 {} ###'.format(fname))

        return feats

    def create_reader(self):
        """reader creator for ctcn model"""
        if self.mode == 'infer':
            return self.make_infer_reader()
        if self.num_threads == 1:
            return self.make_reader()
        else:
            return self.make_multiprocess_reader()

    def make_infer_reader(self):
        """reader for inference"""

        def reader():
            with open(self.filelist) as f:
                reader_list = f.readlines()
            batch_out = []
            for line in reader_list:
                fname = line.strip().split()[0]
                rgb_exist = os.path.exists(
                    os.path.join(self.root, self.rgb, fname + '.pkl'))
                flow_exist = os.path.exists(
                    os.path.join(self.root, self.flow, fname + '.pkl'))
                if not (rgb_exist and flow_exist):
                    logger.info('file not exist', fname)
                    continue
                try:
                    feats = self.load_file(fname)
                    feats, boxes = self.resize(
                        feats, boxes=None, size=self.img_size)
                    h, w = feats.shape[:2]
                    feats = feats.reshape(1, h, w)
                except:
                    logger.info('Error when loading {}'.format(fname))
                    continue
                batch_out.append((feats, fname))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return reader

    def make_reader(self):
        """single process reader"""

        def reader():
            with open(self.filelist) as f:
                reader_list = f.readlines()
            if self.mode == 'train':
                random.shuffle(reader_list)
            fnames = []
            total_boxes = []
            total_labels = []
            total_label_ids = []
            for i in range(len(reader_list)):
                line = reader_list[i]
                splited = line.strip().split()
                rgb_exist = os.path.exists(
                    os.path.join(self.root, self.rgb, splited[0] + '.pkl'))
                flow_exist = os.path.exists(
                    os.path.join(self.root, self.flow, splited[0] + '.pkl'))
                if not (rgb_exist and flow_exist):
                    logger.info('file not exist {}'.format(splited[0]))
                    continue
                fnames.append(splited[0])
                frames_num = int(splited[1]) // self.snippet_length
                num_boxes = int(splited[2])
                box = []
                label = []
                for ii in range(num_boxes):
                    c = splited[3 + 3 * ii]
                    xmin = splited[4 + 3 * ii]
                    xmax = splited[5 + 3 * ii]
                    box.append([
                        float(xmin) / self.snippet_length,
                        float(xmax) / self.snippet_length
                    ])
                    label.append(int(c))
                total_label_ids.append(i)
                total_boxes.append(box)
                total_labels.append(label)
            num_videos = len(fnames)
            batch_out = []
            for idx in range(num_videos):
                fname = fnames[idx]
                try:
                    feats = self.load_file(fname)
                    boxes = copy.deepcopy(total_boxes[idx])
                    labels = copy.deepcopy(total_labels[idx])
                    feats, boxes, labels = self.transform(feats, boxes, labels,
                                                          self.mode)
                    labels = labels.astype('int64')
                    boxes = boxes.astype('float32')
                    num_pos = len(np.where(labels > 0)[0])
                except:
                    logger.info('Error when loading {}'.format(fname))
                    continue
                if (num_pos < 1) and (self.mode == 'train' or
                                      self.mode == 'valid'):
                    #logger.info('=== no pos for ==='.format(fname, num_pos))
                    continue
                if self.mode == 'train' or self.mode == 'valid':
                    batch_out.append((feats, boxes, labels))
                elif self.mode == 'test':
                    batch_out.append(
                        (feats, boxes, labels, total_label_ids[idx]))
                else:
                    raise NotImplementedError('mode {} not implemented'.format(
                        self.mode))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return reader

    def make_multiprocess_reader(self):
        """multiprocess reader"""

        def read_into_queue(reader_list, queue):
            fnames = []
            total_boxes = []
            total_labels = []
            total_label_ids = []
            #for line in reader_list:
            for i in range(len(reader_list)):
                line = reader_list[i]
                splited = line.strip().split()
                rgb_exist = os.path.exists(
                    os.path.join(self.root, self.rgb, splited[0] + '.pkl'))
                flow_exist = os.path.exists(
                    os.path.join(self.root, self.flow, splited[0] + '.pkl'))
                if not (rgb_exist and flow_exist):
                    logger.info('file not exist {}'.format(splited[0]))
                    continue
                fnames.append(splited[0])
                frames_num = int(splited[1]) // self.snippet_length
                num_boxes = int(splited[2])
                box = []
                label = []
                for ii in range(num_boxes):
                    c = splited[3 + 3 * ii]
                    xmin = splited[4 + 3 * ii]
                    xmax = splited[5 + 3 * ii]
                    box.append([
                        float(xmin) / self.snippet_length,
                        float(xmax) / self.snippet_length
                    ])
                    label.append(int(c))
                total_label_ids.append(i)
                total_boxes.append(box)
                total_labels.append(label)
            num_videos = len(fnames)
            batch_out = []
            for idx in range(num_videos):
                fname = fnames[idx]
                try:
                    feats = self.load_file(fname)
                    boxes = copy.deepcopy(total_boxes[idx])
                    labels = copy.deepcopy(total_labels[idx])

                    feats, boxes, labels = self.transform(feats, boxes, labels,
                                                          self.mode)
                    labels = labels.astype('int64')
                    boxes = boxes.astype('float32')
                    num_pos = len(np.where(labels > 0)[0])
                except:
                    logger.info('Error when loading {}'.format(fname))
                    continue
                if (not (num_pos >= 1)) and (self.mode == 'train' or
                                             self.mode == 'valid'):
                    #logger.info('=== no pos for {}, num_pos = {} ==='.format(fname, num_pos))
                    continue

                if self.mode == 'train' or self.mode == 'valid':
                    batch_out.append((feats, boxes, labels))
                elif self.mode == 'test':
                    batch_out.append(
                        (feats, boxes, labels, total_label_ids[idx]))
                else:
                    raise NotImplementedError('mode {} not implemented'.format(
                        self.mode))

                if len(batch_out) == self.batch_size:
                    queue.put(batch_out)
                    batch_out = []
            queue.put(None)

        def queue_reader():
            with open(self.filelist) as f:
                fl = f.readlines()
            if self.mode == 'train':
                random.shuffle(fl)
            n = self.num_threads
            queue_size = 20
            reader_lists = [None] * n
            file_num = int(len(fl) // n)
            for i in range(n):
                if i < len(reader_lists) - 1:
                    tmp_list = fl[i * file_num:(i + 1) * file_num]
                else:
                    tmp_list = fl[i * file_num:]
                reader_lists[i] = tmp_list

            queue = multiprocessing.Queue(queue_size)
            p_list = [None] * len(reader_lists)
            # for reader_list in reader_lists:
            for i in range(len(reader_lists)):
                reader_list = reader_lists[i]
                p_list[i] = multiprocessing.Process(
                    target=read_into_queue, args=(reader_list, queue))
                p_list[i].start()
            reader_num = len(reader_lists)
            finish_num = 0
            while finish_num < reader_num:
                sample = queue.get()
                if sample is None:
                    finish_num += 1
                else:
                    yield sample
            for i in range(len(p_list)):
                if p_list[i].is_alive():
                    p_list[i].join()

        return queue_reader
