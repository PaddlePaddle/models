from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import os
import six
import time
from data_utils import GeneratorEnqueuer

default_config = {
    "shuffle": True,
    "min_resize": 0.5,
    "max_resize": 4,
    "crop_size": 769,
}


def slice_with_pad(a, s, value=0):
    pads = []
    slices = []
    for i in range(len(a.shape)):
        if i >= len(s):
            pads.append([0, 0])
            slices.append([0, a.shape[i]])
        else:
            l, r = s[i]
            if l < 0:
                pl = -l
                l = 0
            else:
                pl = 0
            if r > a.shape[i]:
                pr = r - a.shape[i]
                r = a.shape[i]
            else:
                pr = 0
            pads.append([pl, pr])
            slices.append([l, r])
    slices = list(map(lambda x: slice(x[0], x[1], 1), slices))
    a = a[tuple(slices)]
    a = np.pad(a, pad_width=pads, mode='constant', constant_values=value)
    return a


class CityscapeDataset:
    def __init__(self, dataset_dir, subset='train', config=default_config):
        with open(os.path.join(dataset_dir, subset + '.list'), 'r') as fr:
            file_list = fr.readlines()
        all_images = []
        all_labels = []
        for i in range(len(file_list)):
            img_gt = file_list[i].strip().split(' ')
            all_images.append(os.path.join(dataset_dir, img_gt[0]))
            all_labels.append(os.path.join(dataset_dir, img_gt[1]))

        self.label_files = all_labels
        self.img_files = all_images
        self.index = 0
        self.subset = subset
        self.dataset_dir = dataset_dir
        self.config = config
        self.reset()

    def reset(self, shuffle=False):
        self.index = 0
        if self.config["shuffle"]:
            np.random.shuffle(self.label_files)

    def next_img(self):
        self.index += 1
        if self.index >= len(self.label_files):
            self.reset()

    def get_img(self):
        shape = self.config["crop_size"]
        while True:
            ln = self.label_files[self.index]
            img_name = self.img_files[self.index]
            label = cv2.imread(ln)
            img = cv2.imread(img_name)
            if img is None:
                print("load img failed:", img_name)
                self.next_img()
            else:
                break
        if shape == -1:
            return img, label, ln

        if np.random.rand() > 0.5:
            range_l = 1
            range_r = self.config['max_resize']
        else:
            range_l = self.config['min_resize']
            range_r = 1

        if np.random.rand() > 0.5:
            assert len(img.shape) == 3 and len(
                label.shape) == 3, "{} {}".format(img.shape, label.shape)
            img = img[:, :, ::-1]
            label = label[:, :, ::-1]

        random_scale = np.random.rand(1) * (range_r - range_l) + range_l
        crop_size = int(shape / random_scale)
        bb = crop_size // 2

        def _randint(low, high):
            return int(np.random.rand(1) * (high - low) + low)

        offset_x = np.random.randint(bb, max(bb + 1, img.shape[0] -
                                             bb)) - crop_size // 2
        offset_y = np.random.randint(bb, max(bb + 1, img.shape[1] -
                                             bb)) - crop_size // 2
        img_crop = slice_with_pad(img, [[offset_x, offset_x + crop_size],
                                        [offset_y, offset_y + crop_size]], 128)
        img = cv2.resize(img_crop, (shape, shape))
        label_crop = slice_with_pad(label, [[offset_x, offset_x + crop_size],
                                            [offset_y, offset_y + crop_size]],
                                    255)
        label = cv2.resize(
            label_crop, (shape, shape), interpolation=cv2.INTER_NEAREST)
        return img, label, ln + str(
            (offset_x, offset_y, crop_size, random_scale))

    def get_batch(self, batch_size=1):
        imgs = []
        labels = []
        names = []
        while len(imgs) < batch_size:
            img, label, ln = self.get_img()
            imgs.append(img)
            labels.append(label)
            names.append(ln)
            self.next_img()
        return np.array(imgs), np.array(labels), names

    def get_batch_generator(self,
                            batch_size,
                            total_step,
                            num_workers=8,
                            max_queue=32,
                            use_multiprocessing=True):
        def do_get_batch():
            iter_id = 0
            while True:
                imgs, labels, names = self.get_batch(batch_size)
                labels = labels.astype(np.int32)[:, :, :, 0]
                imgs = imgs[:, :, :, ::-1].transpose(
                    0, 3, 1, 2).astype(np.float32) / (255.0 / 2) - 1
                yield imgs, labels, names
                if not use_multiprocessing:
                    iter_id += 1
                    if iter_id >= total_step:
                        break

        batches = do_get_batch()
        if not use_multiprocessing:
            try:
                from prefetch_generator import BackgroundGenerator
                batches = BackgroundGenerator(batches, 100)
            except:
                print(
                    "You can install 'prefetch_generator' for acceleration of data reading."
                )
            return batches

        def reader():
            try:
                enqueuer = GeneratorEnqueuer(
                    batches, use_multiprocessing=use_multiprocessing)
                enqueuer.start(max_queue_size=max_queue, workers=num_workers)
                generator_out = None
                for i in range(total_step):
                    while enqueuer.is_running():
                        if not enqueuer.queue.empty():
                            generator_out = enqueuer.queue.get()
                            break
                        else:
                            time.sleep(0.02)
                    yield generator_out
                    generator_out = None
                enqueuer.stop()
            finally:
                if enqueuer is not None:
                    enqueuer.stop()

        data_gen = reader()
        return data_gen
