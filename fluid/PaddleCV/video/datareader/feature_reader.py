import sys
from .reader_utils import DataReader
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import random

python_ver = sys.version_info


class FeatureReader(DataReader):
    """
    Data reader for youtube-8M dataset, which was stored as features extracted by prior networks
    This is for the three models: lstm, attention cluster, nextvlad

    dataset cfg: num_classes
                 batch_size
                 list
                 NextVlad only: eigen_file
    """

    def __init__(self, name, phase, cfg):
        self.name = name
        self.phase = phase
        self.num_classes = cfg['num_classes']

        # set batch size and file list
        self.batch_size = cfg['batch_size']
        self.filelist = cfg['list']
        if 'eigen_file' in cfg.keys():
            self.eigen_file = cfg['eigen_file']
        if 'seg_num' in cfg.keys():
            self.seg_num = cfg['seg_num']

    def create_reader(self):
        fl = open(self.filelist).readlines()
        fl = [line.strip() for line in fl if line.strip() != '']
        if self.phase == 'train':
            random.shuffle(fl)

        def reader():
            batch_out = []
            for filepath in fl:
                if python_ver < (3, 0):
                    data = pickle.load(open(filepath, 'rb'))
                else:
                    data = pickle.load(open(filepath, 'rb'), encoding='bytes')
                indexes = list(range(len(data)))
                if self.phase == 'train':
                    random.shuffle(indexes)
                for i in indexes:
                    record = data[i]
                    nframes = record[b'nframes']
                    rgb = record[b'feature'].astype(float)
                    audio = record[b'audio'].astype(float)
                    if self.phase != 'infer':
                        label = record[b'label']
                        one_hot_label = make_one_hot(label, self.num_classes)
                    video = record[b'video']

                    rgb = rgb[0:nframes, :]
                    audio = audio[0:nframes, :]

                    rgb = dequantize(
                        rgb, max_quantized_value=2., min_quantized_value=-2.)
                    audio = dequantize(
                        audio, max_quantized_value=2, min_quantized_value=-2)

                    if self.name == 'NEXTVLAD':
                        # add the effect of eigen values
                        eigen_file = self.eigen_file
                        eigen_val = np.sqrt(np.load(eigen_file)
                                            [:1024, 0]).astype(np.float32)
                        eigen_val = eigen_val + 1e-4
                        rgb = (rgb - 4. / 512) * eigen_val
                    if self.name == 'ATTENTIONCLUSTER':
                        sample_inds = generate_random_idx(rgb.shape[0],
                                                          self.seg_num)
                        rgb = rgb[sample_inds]
                        audio = audio[sample_inds]
                    if self.phase != 'infer':
                        batch_out.append((rgb, audio, one_hot_label))
                    else:
                        batch_out.append((rgb, audio, video))
                    if len(batch_out) == self.batch_size:
                        yield batch_out
                        batch_out = []

        return reader


def dequantize(feat_vector, max_quantized_value=2., min_quantized_value=-2.):
    """
    Dequantize the feature from the byte format to the float format
    """

    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value

    return feat_vector * scalar + bias


def make_one_hot(label, dim=3862):
    one_hot_label = np.zeros(dim)
    one_hot_label = one_hot_label.astype(float)
    for ind in label:
        one_hot_label[int(ind)] = 1
    return one_hot_label


def generate_random_idx(feature_len, seg_num):
    idxs = []
    stride = float(feature_len) / seg_num
    for i in range(seg_num):
        pos = (i + np.random.random()) * stride
        idxs.append(min(feature_len - 1, int(pos)))
    return idxs
