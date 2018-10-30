import os
import math
import random
import functools
import numpy as np
import paddle
from PIL import Image, ImageEnhance

random.seed(0)

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 1024000

DATA_DIR = "./data/"
TRAIN_LIST = './data/CUB200_train.txt'
TEST_LIST = './data/CUB200_val.txt'
#DATA_DIR = "./data/CUB200/"
#TRAIN_LIST = './data/CUB200/CUB200_train.txt'
#TEST_LIST = './data/CUB200/CUB200_val.txt'
train_data = {}
test_data = {}
train_list = open(TRAIN_LIST, "r").readlines()
train_image_list = []
for i, item in enumerate(train_list):
    path, label = item.strip().split()
    label = int(label) - 1
    train_image_list.append((path, label))
    if label not in train_data:
        train_data[label] = []
    train_data[label].append(path)

test_list = open(TEST_LIST, "r").readlines()
test_image_list = []
infer_image_list = []
for i, item in enumerate(test_list):
    path, label = item.strip().split()
    label = int(label) - 1
    test_image_list.append((path, label))
    infer_image_list.append(path)
    if label not in test_data:
        test_data[label] = []
    test_data[label].append(path)

print("train_data size:", len(train_data))
print("test_data size:", len(test_data))
print("test_data image number:", len(test_image_list))
random.shuffle(test_image_list)


img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.BILINEAR)
    return img

def Scale(img, size):
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), Image.BILINEAR)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), Image.BILINEAR)

def CenterCrop(img, size):
    w, h = img.size
    th, tw = int(size), int(size)
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return img.crop((x1, y1, x1 + tw, y1 + th))

def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img

def RandomResizedCrop(img, size):
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.08, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x1 = random.randint(0, img.size[0] - w)
            y1 = random.randint(0, img.size[1] - h)

            img = img.crop((x1, y1, x1 + w, y1 + h))
            assert(img.size == (w, h))

            return img.resize((size, size), Image.BILINEAR)

    w = min(img.size[0], img.size[1])
    i = (img.size[1] - w) // 2
    j = (img.size[0] - w) // 2
    img = img.crop((i, j, i+w, j+w))
    img = img.resize((size, size), Image.BILINEAR)
    return img


def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * random.uniform(scale_min,
                                                             scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = random.randint(0, img.size[0] - w)
    j = random.randint(0, img.size[1] - h)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size, size), Image.BILINEAR)
    return img


def rotate_image(img):
    angle = random.randint(-10, 10)
    img = img.rotate(angle)
    return img


def distort_color(img):
    def random_brightness(img, lower=0.8, upper=1.2):
        e = random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.8, upper=1.2):
        e = random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.8, upper=1.2):
        e = random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img

def process_image_imagepath(sample, mode, color_jitter, rotate):
    imgpath = sample[0]
    img = Image.open(imgpath)
    if mode == 'train':
        if rotate: img = rotate_image(img)
        img = RandomResizedCrop(img, DATA_DIM)
    else:
        img = Scale(img, 256)
        img = CenterCrop(img, DATA_DIM)
    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if random.randint(0, 1) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    if mode in ['train', 'test']:
        return img, sample[1]
    elif mode == 'infer':
        return [img]


def eml_iterator(data,
                 mode,
                 batch_size,
                 samples_each_class,
                 iter_size,
                 shuffle=False,
                 color_jitter=False,
                 rotate=False):
    def reader():
        labs = list(data.keys())
        lab_num = len(labs)
        ind = list(range(0, lab_num))
        assert batch_size % samples_each_class == 0, "batch_size % samples_each_class != 0"
        num_class = batch_size // samples_each_class
        for i in range(iter_size):
            random.shuffle(ind)
            for n in range(num_class):
                lab_ind = ind[n]
                label = labs[lab_ind]
                data_list = data[label]
                random.shuffle(data_list)
                for s in range(samples_each_class):
                    path = DATA_DIR + data_list[s]
                    yield path, label

    mapper = functools.partial(
        process_image_imagepath, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE, order=True)


def quadruplet_iterator(data,
                        mode,
                        class_num,
                        samples_each_class,
                        iter_size,
                        shuffle=False,
                        color_jitter=False,
                        rotate=False):
    def reader():
        labs = list(data.keys())
        lab_num = len(labs)
        ind = list(range(0, lab_num))
        for i in range(iter_size):
            random.shuffle(ind)
            ind_sample = ind[:class_num]

            for ind_i in ind_sample:
                lab = labs[ind_i]
                data_list = data[lab]
                data_ind = list(range(0, len(data_list)))
                random.shuffle(data_ind)
                anchor_ind = data_ind[:samples_each_class]

                for anchor_ind_i in anchor_ind:
                    anchor_path = DATA_DIR + data_list[anchor_ind_i]
                    yield anchor_path, lab

    mapper = functools.partial(
        process_image_imagepath, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE, order=True)


def triplet_iterator(data,
                     mode,
                     batch_size,
                     iter_size,
                     shuffle=False,
                     color_jitter=False,
                     rotate=False):
    def reader():
        labs = list(data.keys())
        lab_num = len(labs)
        ind = list(range(0, lab_num))
        for i in range(iter_size):
            random.shuffle(ind)
            ind_pos, ind_neg = ind[:2]
            lab_pos = labs[ind_pos]
            pos_data_list = data[lab_pos]
            data_ind = list(range(0, len(pos_data_list)))
            random.shuffle(data_ind)
            anchor_ind, pos_ind = data_ind[:2]

            lab_neg = labs[ind_neg]
            neg_data_list = data[lab_neg]
            neg_ind = random.randint(0, len(neg_data_list) - 1)
            
            anchor_path = DATA_DIR + pos_data_list[anchor_ind]
            yield anchor_path, lab_pos

            pos_path = DATA_DIR + pos_data_list[pos_ind]
            yield pos_path, lab_pos

            neg_path = DATA_DIR + neg_data_list[neg_ind]
            yield neg_path, lab_neg


    mapper = functools.partial(
        process_image_imagepath, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE, order=True)


def image_iterator(data,
                   mode,
                   shuffle=False,
                   color_jitter=False,
                   rotate=False):
    def test_reader():
        for i in range(len(data)):
            path, label = data[i]
            path = DATA_DIR + path 
            yield path, label

    def infer_reader():
        for i in range(len(data)):
            path = data[i]
            path = DATA_DIR + path 
            yield [path]

    if mode == "test":
        mapper = functools.partial(
            process_image_imagepath, mode=mode, color_jitter=color_jitter, rotate=rotate)
        return paddle.reader.xmap_readers(mapper, test_reader, THREAD, BUF_SIZE)
    elif mode == "infer":
        mapper = functools.partial(
            process_image_imagepath, mode=mode, color_jitter=color_jitter, rotate=rotate)
        return paddle.reader.xmap_readers(mapper, infer_reader, THREAD, BUF_SIZE)


def eml_train(batch_size, samples_each_class):
    return eml_iterator(train_data, 'train', batch_size, samples_each_class, iter_size = 100, \
                           shuffle=True, color_jitter=False, rotate=False)

def quadruplet_train(class_num, samples_each_class):
    return quadruplet_iterator(train_data, 'train', class_num, samples_each_class, iter_size=100, \
                           shuffle=True, color_jitter=False, rotate=False)
            
def triplet_train(batch_size):
    assert(batch_size % 3 == 0)
    return triplet_iterator(train_data, 'train', batch_size, iter_size = batch_size//3 * 100, \
                           shuffle=True, color_jitter=False, rotate=False)

def test():
    return image_iterator(test_image_list, "test", shuffle=False)

def infer():
    return image_iterator(infer_image_list, "infer", shuffle=False)
