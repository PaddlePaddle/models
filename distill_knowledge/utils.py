"""
    This script contains the dataset loader
"""
import numpy as np


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class Dataset(object):
    def __init__(self,
                 images,
                 labels,
                 soft_labels=None,
                 one_hot=False,
                 reshape=False,
                 seed=123):
        np.random.seed(seed)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        self.images = images
        self.labels = labels
        self.soft_labels = soft_labels
        if reshape:
            self.images = self.images.reshape(images.shape[0], 1, 28, 28)
        if one_hot:
            self.labels = dense_to_one_hot(self.labels, 10)

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples or self._index_in_epoch == batch_size and shuffle:
            # print("Data Shuffling")
            self._epochs_completed += 1
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self.images = self.images[perm0]
            self.labels = self.labels[perm0]
            if self.soft_labels is not None:
                # print("soft shape", self.soft_labels.shape)
                self.soft_labels = self.soft_labels[perm0]

            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        img_batch = self.images[start:end]
        label_batch = self.labels[start:end]
        if self.soft_labels is not None:
            soft_label_batch = self.soft_labels[start:end]
            return list(zip(img_batch, label_batch, soft_label_batch))
        else:
            # print(img_batch.shape, label_batch.shape)
            return list(zip(img_batch, label_batch))


def read_data_sets(is_soft=False, one_hot=False, reshape=False, temp=str(3.0)):
    if is_soft:
        data_dir = './data/mnist_soft_{}.npz'.format(temp)
    else:
        data_dir = './data/mnist.npz'
    print("Loading ", data_dir)
    mnist_data = np.load(data_dir)
    train_x = mnist_data['train_x']
    train_y = mnist_data['train_y']
    test_x = mnist_data['test_x']
    test_y = mnist_data['test_y']
    if is_soft:
        train_y_soft = mnist_data['train_y_soft']
        test_y_soft = np.zeros(shape=(test_x.shape[0], 10))
        train_set = Dataset(
            train_x,
            train_y,
            soft_labels=train_y_soft,
            one_hot=one_hot,
            reshape=reshape)
        test_set = Dataset(
            test_x,
            test_y,
            soft_labels=test_y_soft,
            one_hot=one_hot,
            reshape=reshape)
        # test_set = Dataset(test_x, test_y, one_hot=one_hot, reshape=reshape)
    else:
        train_set = Dataset(train_x, train_y, one_hot=one_hot, reshape=reshape)
        test_set = Dataset(test_x, test_y, one_hot=one_hot, reshape=reshape)
    return train_set, test_set
