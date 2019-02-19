import os

import numpy as np
import math
import random
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data.sampler import Sampler
import torchvision
import pickle
from tqdm import tqdm
import time
import multiprocessing

TRAINER_NUMS = int(os.getenv("PADDLE_TRAINER_NUM", "1"))
TRAINER_ID = int(os.getenv("PADDLE_TRAINER_ID", "0"))
epoch = 0

FINISH_EVENT = "FINISH_EVENT"
#def paddle_data_loader(torch_dataset, indices=None, concurrent=1, queue_size=3072, use_uint8_reader=False):
class PaddleDataLoader(object):
    def __init__(self, torch_dataset, indices=None, concurrent=16, queue_size=3072):
        self.torch_dataset = torch_dataset
        self.data_queue = multiprocessing.Queue(queue_size)
        self.indices = indices
        self.concurrent = concurrent

    def _worker_loop(self, dataset, worker_indices, worker_id):
        cnt = 0
        for idx in worker_indices:
            cnt += 1
            img, label = self.torch_dataset[idx]
            img = np.array(img).astype('uint8').transpose((2, 0, 1))
            self.data_queue.put((img, label))
        print("worker: [%d] read [%d] samples. " % (worker_id, cnt))
        self.data_queue.put(FINISH_EVENT)

    def reader(self):
        def _reader_creator():
            worker_processes = []
            total_img = len(self.torch_dataset)
            print("total image: ", total_img)
            if self.indices is None:
                self.indices = [i for i in xrange(total_img)]
                random.seed(time.time())
                random.shuffle(self.indices)
                print("shuffle indices: %s ..." % self.indices[:10])

            imgs_per_worker = int(math.ceil(total_img / self.concurrent))
            for i in xrange(self.concurrent):
                start = i * imgs_per_worker
                end = (i + 1) * imgs_per_worker if i != self.concurrent - 1 else None
                sliced_indices = self.indices[start:end]
                w = multiprocessing.Process(
                    target=self._worker_loop,
                    args=(self.torch_dataset, sliced_indices, i)
                )
                w.daemon = True
                w.start()
                worker_processes.append(w)
            finish_workers = 0
            worker_cnt = len(worker_processes)
            while finish_workers < worker_cnt:
                sample = self.data_queue.get()
                if sample == FINISH_EVENT:
                    finish_workers += 1
                else:
                    yield sample

        return _reader_creator

def train(traindir, sz, min_scale=0.08):
    train_tfms = [
        transforms.RandomResizedCrop(sz, scale=(min_scale, 1.0)),
        transforms.RandomHorizontalFlip()
    ]
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(train_tfms))
    return PaddleDataLoader(train_dataset).reader()

def test(valdir, bs, sz, rect_val=False):
    if rect_val:
        idx_ar_sorted = sort_ar(valdir)
        idx_sorted, _ = zip(*idx_ar_sorted)
        idx2ar = map_idx2ar(idx_ar_sorted, bs)

        ar_tfms = [transforms.Resize(int(sz* 1.14)), CropArTfm(idx2ar, sz)]
        val_dataset = ValDataset(valdir, transform=ar_tfms)
        return PaddleDataLoader(val_dataset, concurrent=1, indices=idx_sorted).reader()

    val_tfms = [transforms.Resize(int(sz* 1.14)), transforms.CenterCrop(sz)]
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose(val_tfms))

    return PaddleDataLoader(val_dataset).reader()



def create_validation_set(valdir, batch_size, target_size, rect_val, distributed):
    print("create_validation_set", valdir, batch_size, target_size, rect_val, distributed)
    if rect_val:
        idx_ar_sorted = sort_ar(valdir)
        idx_sorted, _ = zip(*idx_ar_sorted)
        idx2ar = map_idx2ar(idx_ar_sorted, batch_size)

        ar_tfms = [transforms.Resize(int(target_size * 1.14)), CropArTfm(idx2ar, target_size)]
        val_dataset = ValDataset(valdir, transform=ar_tfms)
        val_sampler = DistValSampler(idx_sorted, batch_size=batch_size, distributed=distributed)
        return val_dataset, val_sampler

    val_tfms = [transforms.Resize(int(target_size * 1.14)), transforms.CenterCrop(target_size)]
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose(val_tfms))
    val_sampler = DistValSampler(list(range(len(val_dataset))), batch_size=batch_size, distributed=distributed)
    return val_dataset, val_sampler


class ValDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(ValDataset, self).__init__(root, transform, target_transform)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            for tfm in self.transform:
                if isinstance(tfm, CropArTfm):
                    sample = tfm(sample, index)
                else:
                    sample = tfm(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DistValSampler(Sampler):
    # DistValSampler distrbutes batches equally (based on batch size) to every gpu (even if there aren't enough images)
    # WARNING: Some baches will contain an empty array to signify there aren't enough images
    # Distributed=False - same validation happens on every single gpu
    def __init__(self, indices, batch_size, distributed=True):
        self.indices = indices
        self.batch_size = batch_size
        if distributed:
            self.world_size = TRAINER_NUMS
            self.global_rank = TRAINER_ID
        else:
            self.global_rank = 0
            self.world_size = 1

        # expected number of batches per sample. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_batches = int(math.ceil(len(self.indices) / self.world_size / self.batch_size))

        # num_samples = total images / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_batches * self.batch_size

    def __iter__(self):
        offset = self.num_samples * self.global_rank
        sampled_indices = self.indices[offset:offset + self.num_samples]
        print("DistValSampler: self.world_size: ", self.world_size, " self.global_rank: ", self.global_rank)
        for i in range(self.expected_num_batches):
            offset = i * self.batch_size
            yield sampled_indices[offset:offset + self.batch_size]

    def __len__(self):
        return self.expected_num_batches

    def set_epoch(self, epoch):
        return


class CropArTfm(object):
    def __init__(self, idx2ar, target_size):
        self.idx2ar, self.target_size = idx2ar, target_size

    def __call__(self, img, idx):
        target_ar = self.idx2ar[idx]
        if target_ar < 1:
            w = int(self.target_size / target_ar)
            size = (w // 8 * 8, self.target_size)
        else:
            h = int(self.target_size * target_ar)
            size = (self.target_size, h // 8 * 8)
        return torchvision.transforms.functional.center_crop(img, size)


def sort_ar(valdir):
    idx2ar_file = valdir + '/../sorted_idxar.p'
    if os.path.isfile(idx2ar_file):
        return pickle.load(open(idx2ar_file, 'rb'))
    print('Creating AR indexes. Please be patient this may take a couple minutes...')
    val_dataset = datasets.ImageFolder(valdir)  # AS: TODO: use Image.open instead of looping through dataset
    sizes = [img[0].size for img in tqdm(val_dataset, total=len(val_dataset))]
    idx_ar = [(i, round(s[0] * 1.0/ s[1], 5)) for i, s in enumerate(sizes)]
    sorted_idxar = sorted(idx_ar, key=lambda x: x[1])
    pickle.dump(sorted_idxar, open(idx2ar_file, 'wb'))
    print('Done')
    return sorted_idxar

def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def map_idx2ar(idx_ar_sorted, batch_size):
    ar_chunks = list(chunks(idx_ar_sorted, batch_size))
    idx2ar = {}
    for chunk in ar_chunks:
        idxs, ars = list(zip(*chunk))
        mean = round(np.mean(ars), 5)
        for idx in idxs:
            idx2ar[idx] = mean
    return idx2ar

if __name__ == "__main__":
    #ds, sampler = create_validation_set("/data/imagenet/validation", 128, 288, True, True)
    #for item in sampler:
    #    for idx in item:
    #        ds[idx]

    import time
    test_reader = test(valdir="/data/imagenet/validation", bs=50, sz=288, rect_val=True)
    start_ts = time.time()
    for idx, data in enumerate(test_reader()):
        print(idx, data[0].shape, data[1])
        if idx == 10:
            break
        if (idx + 1) % 1000 == 0:
            cost = (time.time() - start_ts)
            print("%d samples per second" % (1000 / cost))
            start_ts = time.time()