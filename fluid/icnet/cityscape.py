import os
import cv2
import numpy as np
import paddle.v2 as paddle

DATA_PATH = "../../data/cityscape"
TRAIN_LIST = DATA_PATH + "/train.list"
TEST_LIST = DATA_PATH + "/val.list"
IGNORE_LABEL=255
NUM_CLASSES=19
TRAIN_DATA_SHAPE=(3, 720, 720)
TEST_DATA_SHAPE=(3, 1024, 2048)
IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

def train_data_shape():
	return TRAIN_DATA_SHAPE

def test_data_shape():
	return TEST_DATA_SHAPE

def num_classes():
	return NUM_CLASSES

class DataGenerater:
    def __init__(self, data_list):
        self.image_label = []
        with open(data_list, 'r') as f:
            for line in f:
                image_file, label_file = line.strip().split(' ')
                self.image_label.append((image_file, label_file))

    def create_reader(self):
        def reader():
            for image, label in self.image_label:
                yield image, label
        return reader

def load(image_label):
    image, label = image_label
    image = paddle.image.load_image(DATA_PATH + "/" + image, is_color=True).astype("float32")
    image -= IMG_MEAN
    label = paddle.image.load_image(DATA_PATH + "/" + label, is_color=False).astype("float32")
    return image, label

def flip(image_label):
    image, label = image_label
    r = np.random.rand(1)
    if r > 0.5:
        image = paddle.image.left_right_flip(image, is_color=True)
        label = paddle.image.left_right_flip(label, is_color=False)
    return image, label

def scaling(image_label):
    image, label = image_label
    scale = np.random.uniform(0.5,2.0,1)[0]
    h_new = int(image.shape[0] * scale)
    w_new = int(image.shape[1] * scale)
    image = cv2.resize(image, (w_new, h_new))
    label = cv2.resize(label, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    return image, label

def padding_as(image, h, w, is_color):
    pad_h = max(image.shape[0], h) - image.shape[0]
    pad_w = max(image.shape[1], w) - image.shape[1]
    if is_color:
        return np.pad(image, ((0, pad_h), (0, pad_w), (0,0)), 'constant')
    else:
        return np.pad(image, ((0, pad_h), (0, pad_w)), 'constant')


def resize(image_label):
    ignore_label = IGNORE_LABEL
    image, label = image_label
    label = label - ignore_label
    if len(label.shape) == 2:
        label = label[:,:,np.newaxis]
    combined = np.concatenate((image, label), axis=2)
    crop_h = TRAIN_DATA_SHAPE[1]
    crop_w = TRAIN_DATA_SHAPE[2]
    combined = padding_as(combined, crop_h, crop_w, is_color=True)
    combined = paddle.image.random_crop(combined, crop_h, is_color=True)
    image = combined[:,:,0:3]
    label = combined[:,:,3:4] + ignore_label
    return image, label

def to_chw(image_label):
    return paddle.image.to_chw(image_label[0]), paddle.image.to_chw(image_label[1]), paddle.image.to_chw(image_label[2]), paddle.image.to_chw(image_label[3])

def scale(image_label):
    image, label = image_label
    label = label.astype("float32")

    h = label.shape[0] / 4
    w = label.shape[1] / 4
    label_sub1 = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis]

    h = label.shape[0] / 8
    w = label.shape[1] / 8
    label_sub2 = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis]

    h = label.shape[0] / 16
    w = label.shape[1] / 16
    label_sub4 = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis]

    return image, label_sub1, label_sub2, label_sub4

def batch_reader(batch_size, reader):
    def b_reader():
        images = []
        labels_sub1 = []
        labels_sub2 = []
        labels_sub4 = []
        count = 0
        for image, label_sub1, label_sub2, label_sub4 in reader():
            count+=1
            images.append(image)
            labels_sub1.append(label_sub1)
            labels_sub2.append(label_sub2)
            labels_sub4.append(label_sub4)
            if count == batch_size:
                yield np.array(images), np.array(labels_sub1), np.array(labels_sub2), np.array(labels_sub4)
                images = []
                labels_sub1 = []
                labels_sub2 = []
                labels_sub4 = []
		count = 0
        if images:
            yield np.array(images), np.array(labels_sub1), np.array(labels_sub2), np.array(labels_sub4)
    return b_reader


def mask(image_labels):
    label_sub1 = image_labels[1]
    label_sub2 = image_labels[2]
    label_sub4 = image_labels[3]
    mask_sub1 = np.where(((label_sub1 < (NUM_CLASSES+1)) & (label_sub1 != IGNORE_LABEL)).flatten())[0]
    mask_sub2 = np.where(((label_sub2 < (NUM_CLASSES+1)) & (label_sub2 != IGNORE_LABEL)).flatten())[0]
    mask_sub4 = np.where(((label_sub4 < (NUM_CLASSES+1)) & (label_sub4 != IGNORE_LABEL)).flatten())[0]
    return image_labels[0].astype("float32"), label_sub1,  mask_sub1.astype("int32"), label_sub2, mask_sub2.astype("int32"), label_sub4, mask_sub4.astype("int32")


def train(batch_size=32, random_mirror=False, random_scaling=False):
    reader = DataGenerater(TRAIN_LIST).create_reader()
    reader = paddle.reader.shuffle(reader, 10000)
    reader = paddle.reader.map_readers(load, reader)
    if random_mirror:
        reader = paddle.reader.map_readers(flip, reader)
    if random_scaling:
        reader = paddle.reader.map_readers(scaling, reader)
    reader = paddle.reader.map_readers(resize, reader)
    reader = paddle.reader.map_readers(scale, reader)
    reader = paddle.reader.map_readers(to_chw, reader)
    reader = batch_reader(batch_size, reader)
    reader = paddle.reader.map_readers(mask, reader)
    return reader

def test_process(img_label):
    img = paddle.image.to_chw(img_label[0])[np.newaxis,:]
    label = img_label[1][np.newaxis,:,:,np.newaxis]
    label_mask = np.where((label != IGNORE_LABEL).flatten())[0]
    return img, label.astype("float32"), label_mask.astype("int32") 

def test():
    reader = DataGenerater(TEST_LIST).create_reader()
    reader =  paddle.reader.map_readers(load, reader)
    reader =  paddle.reader.map_readers(test_process, reader)
    return reader

def infer(image_list):
    reader = DataGenerater(TEST_LIST).create_reader()
    
