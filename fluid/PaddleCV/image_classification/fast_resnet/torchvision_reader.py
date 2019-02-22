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
epoch = 0

FINISH_EVENT = "FINISH_EVENT"
class PaddleDataLoader(object):
    def __init__(self, torch_dataset, indices=None, concurrent=4, queue_size=1024, shuffle_seed=None, is_train=True):
        self.torch_dataset = torch_dataset
        self.data_queue = multiprocessing.Queue(queue_size)
        self.indices = indices
        self.concurrent = concurrent
        self.shuffle_seed = shuffle_seed
        self.is_train = is_train

    def _shuffle_worker_indices(self, indices, shuffle_seed = None):
        import copy
        shuffled_indices = copy.deepcopy(indices)
        random.seed(time.time() if shuffle_seed is None else shuffle_seed)
        random.shuffle(shuffled_indices)
        sampels_per_worker = len(shuffled_indices) / TRAINER_NUMS
        start = TRAINER_ID * sampels_per_worker
        end = (TRAINER_ID + 1) * sampels_per_worker
        ret = shuffled_indices[start:end]
        print("shuffling worker indices trainer_id: [%d], num_trainers:[%d], len: [%d], start: [%d], end: [%d]" % (TRAINER_ID, TRAINER_NUMS, len(ret), start, end))
        return ret
        
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
                if self.is_train:
                    print("shuffle indices by seed: ", self.shuffle_seed)
                    self.indices = self._shuffle_worker_indices(self.indices, self.shuffle_seed)
                print("samples: %d shuffled indices: %s ..." % (len(self.indices), self.indices[:10]))

            imgs_per_worker = int(math.ceil(len(self.indices) / self.concurrent))
            for i in xrange(self.concurrent):
                start = i * imgs_per_worker
                end = (i + 1) * imgs_per_worker if i != self.concurrent - 1 else -1
                print("loader thread: [%d] start idx: [%d], end idx: [%d]" % (i, start, end))
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

def train(traindir, sz, min_scale=0.08, shuffle_seed=None):
    train_tfms = [
        transforms.RandomResizedCrop(sz, scale=(min_scale, 1.0)),
        transforms.RandomHorizontalFlip()
    ]
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(train_tfms))
    return PaddleDataLoader(train_dataset, shuffle_seed=shuffle_seed)

def test(valdir, bs, sz, rect_val=False):
    if rect_val:
        idx_ar_sorted = sort_ar(valdir)
        idx_sorted, _ = zip(*idx_ar_sorted)
        idx2ar = map_idx2ar(idx_ar_sorted, bs)

        ar_tfms = [transforms.Resize(int(sz* 1.14)), CropArTfm(idx2ar, sz)]
        val_dataset = ValDataset(valdir, transform=ar_tfms)
        return PaddleDataLoader(val_dataset, concurrent=1, indices=idx_sorted, is_train=False)

    val_tfms = [transforms.Resize(int(sz* 1.14)), transforms.CenterCrop(sz)]
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose(val_tfms))

    return PaddleDataLoader(val_dataset, is_train=False)


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
    #ds, sampler = create_validation_set("/data/imagenet/validation", 128, 288, True, True)
    #for item in sampler:
    #    for idx in item:
    #        ds[idx]

    import time
    test_reader = test(valdir="/data/imagenet/validation", bs=50, sz=288, rect_val=True)
    start_ts = time.time()
    for idx, data in enumerate(test_reader.reader()):
        print(idx, data[0].shape, data[1])
        if idx == 10:
            break
        if (idx + 1) % 1000 == 0:
            cost = (time.time() - start_ts)
            print("%d samples per second" % (1000 / cost))
            start_ts = time.time()