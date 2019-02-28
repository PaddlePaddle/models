import os

import numpy as np
import math
import random
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pickle
from tqdm import tqdm
import time
import multiprocessing

TRAINER_NUMS = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
TRAINER_ID = int(os.getenv("PADDLE_TRAINER_ID", "0"))

FINISH_EVENT = "FINISH_EVENT"
class PaddleDataLoader(object):
    def __init__(self, torch_dataset, indices=None, concurrent=16, queue_size=1024, shuffle=True, batch_size=224, is_distributed=True):
        self.torch_dataset = torch_dataset
        self.data_queue = multiprocessing.Queue(queue_size)
        self.indices = indices
        self.concurrent = concurrent
        self.shuffle_seed = 0
        self.shuffle = shuffle
        self.is_distributed = is_distributed
        self.batch_size = batch_size

    def _worker_loop(self, dataset, worker_indices, worker_id):
        cnt = 0
        print("worker [%d], len: [%d], indices: [%s]"%(worker_id, len(worker_indices), worker_indices[:10]))
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
            if self.shuffle:
                random.seed(self.shuffle_seed)
                random.shuffle(self.indices)
            worker_indices = self.indices
            if self.is_distributed:
                cnt_per_node = len(self.indices) / TRAINER_NUMS
                offset = TRAINER_ID * cnt_per_node
                worker_indices = self.indices[offset: (offset + cnt_per_node)]
            if len(worker_indices) % self.batch_size != 0:
                worker_indices += worker_indices[:(self.batch_size - (len(worker_indices) % self.batch_size))]
            print("shuffle: [%d], shuffle seed: [%d], worker indices len: [%d], %s" % (self.shuffle, self.shuffle_seed, len(worker_indices), worker_indices[:10]))

            cnt_per_thread = int(math.ceil(len(worker_indices) / self.concurrent))
            for i in xrange(self.concurrent):
                offset = i * cnt_per_thread
                thread_incides = worker_indices[offset: (offset + cnt_per_thread)]
                print("loader thread: [%d] start idx: [%d], end idx: [%d], len: [%d]" % (i, offset, (offset + cnt_per_thread), len(thread_incides)))
                w = multiprocessing.Process(
                    target=self._worker_loop,
                    args=(self.torch_dataset, thread_incides, i)
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

def train(traindir, bs, sz, min_scale=0.08):
    train_tfms = [
        transforms.RandomResizedCrop(sz, scale=(min_scale, 1.0)),
        transforms.RandomHorizontalFlip()
    ]
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(train_tfms))
    return PaddleDataLoader(train_dataset, batch_size=bs)

def test(valdir, bs, sz, rect_val=False):
    if rect_val:
        idx_ar_sorted = sort_ar(valdir)
        idx_sorted, _ = zip(*idx_ar_sorted)
        idx2ar = map_idx2ar(idx_ar_sorted, bs)

        ar_tfms = [transforms.Resize(int(sz* 1.14)), CropArTfm(idx2ar, sz)]
        val_dataset = ValDataset(valdir, transform=ar_tfms)
        return PaddleDataLoader(val_dataset, concurrent=1, indices=idx_sorted, shuffle=False, is_distributed=False)

    val_tfms = [transforms.Resize(int(sz* 1.14)), transforms.CenterCrop(sz)]
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose(val_tfms))

    return PaddleDataLoader(val_dataset, is_distributed=False)


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
        return transforms.functional.center_crop(img, size)


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
    reader = test("/work/fast_resnet_data", 64, 128).reader()
    print(next(reader()))