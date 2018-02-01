#by zhxfl 2018.01.24
""" @package docstring
load speech data from disk
"""

import random
import Queue
import numpy
import struct
import data_utils.trans_mean_variance_norm as trans_mean_variance_norm
import data_utils.trans_add_delta as trans_add_delta

g_lblock = []
g_que_sample = Queue.Queue()
g_nframe_dim = 120 * 11
g_nstart_block_idx = 0
g_nload_block_num = 1
g_ndrop_frame_len = 256

class OneBlock(object):
    """ Documentation for a class.
        struct for one block :
        contain label, label desc, feature, feature_desc
    """
    def __init__(self):
        """The constructor."""
        self.label = ""
        self.label_desc = ""
        self.feature = ""
        self.feature_desc = ""

def set_trans(ltrans):
    global g_ltrans
    g_ltrans = ltrans

def load_list(sfeature_lst, slabel_lst):
    """ load list """
    global g_lblock 

    lFeature = open(sfeature_lst).readlines()
    lLabel = open(slabel_lst).readlines()
    assert len(lLabel) == len(lFeature)
    for i in range(0, len(lFeature), 2):
        one_block = OneBlock()

        one_block.label = lLabel[i]
        one_block.label_desc = lLabel[i + 1]
        one_block.feature = lFeature[i]
        one_block.feature_desc = lFeature[i + 1]
        g_lblock.append(one_block)
    
    random.shuffle(g_lblock)

def load_one_block(lsample, id):
    """read one block"""
    global g_lblock
    if id >= len(g_lblock):
        return

    slabel_path = g_lblock[id].label.replace("\n", "")
    slabel_desc_path = g_lblock[id].label_desc.replace("\n", "")
    sfeature_path = g_lblock[id].feature.replace("\n", "")
    sfeature_desc_path = g_lblock[id].feature_desc.replace("\n", "")

    llabel_line = open(slabel_desc_path).readlines()
    lfeature_line = open(sfeature_desc_path).readlines()
    
    file_lable_bin = open(slabel_path, "r")
    file_feature_bin = open(sfeature_path, "r")
    
    sample_num = int(llabel_line[0].split()[1])
    assert sample_num == int(lfeature_line[0].split()[1])

    llabel_line = llabel_line[1:]
    lfeature_line = lfeature_line[1:]

    for i in range(sample_num):
        # read label 
        llabel_split = llabel_line[i].split()
        nlabel_start = int(llabel_split[2])
        nlabel_size = int(llabel_split[3])
        nlabel_frame_num = int(llabel_split[4])

        file_lable_bin.seek(nlabel_start, 0)
        label_bytes = file_lable_bin.read(nlabel_size)
        assert nlabel_frame_num * 4 == len(label_bytes)
        label_array = struct.unpack('I' * nlabel_frame_num, label_bytes)
        label_data = numpy.array(label_array, dtype=int)
        label_data = label_data.reshape((nlabel_frame_num, 1))

        # read feature
        lfeature_split = lfeature_line[i].split()
        nfeature_start = int(lfeature_split[2])
        nfeature_size = int(lfeature_split[3])
        nfeature_frame_num = int(lfeature_split[4]) 
        nfeature_frame_dim = int(lfeature_split[5])

        file_feature_bin.seek(nfeature_start, 0)
        feature_bytes = file_feature_bin.read(nfeature_size)
        assert nfeature_frame_num * nfeature_frame_dim * 4 == len(feature_bytes)
        feature_array = struct.unpack('f' * nfeature_frame_num * nfeature_frame_dim, feature_bytes)
        feature_data = numpy.array(feature_array, dtype=float)
        feature_data = feature_data.reshape((nfeature_frame_num, nfeature_frame_dim))
        global g_ndrop_frame_len
        #drop long sentence
        if g_ndrop_frame_len < feature_data.shape[0]:
            continue
        lsample.append((feature_data, label_data))

def load_block(lblock_id):
    """
        read blocks
    """
    global g_ltrans
    lsample = []
    for id in lblock_id:
        load_one_block(lsample, id)
    
    # transform sample
    for (nidx, sample) in enumerate(lsample):
        for trans in g_ltrans:
            sample = trans.perform_trans(sample)
        print nidx
        lsample[nidx] = sample

    return lsample

def move_sample(lsample):
    """
        move sample to queue
    """
    # random
    random.shuffle(lsample)

    global g_que_sample
    for sample in lsample:
        g_que_sample.put(sample)

def get_one_batch(nbatch_size):
    """
        construct one batch 
    """
    global g_que_sample
    global g_nstart_block_idx
    global g_nframe_dim
    global g_nload_block_num
    if g_que_sample.empty():
        lsample = load_block(range(g_nstart_block_idx, g_nstart_block_idx + g_nload_block_num, 1))
        move_sample(lsample)
        g_nstart_block_idx += g_nload_block_num

    if g_que_sample.empty():
        g_nstart_block_idx = 0
        return None
    #cal all frame num
    ncur_len = 0
    lod = [0]
    samples = [] 
    bat_feature = numpy.zeros((nbatch_size, g_nframe_dim))
    for i in range(nbatch_size):
        # empty clear zero 
        if g_que_sample.empty():
            g_nstart_block_idx = 0
        # copy
        else:
            (one_feature, one_label) = g_que_sample.get()
            samples.append((one_feature, one_label))
            ncur_len += one_feature.shape[0]
            lod.append(ncur_len)
    
    bat_feature = numpy.zeros((ncur_len, g_nframe_dim), dtype="float32")
    bat_label = numpy.zeros((ncur_len, 1), dtype="int64")
    ncur_len = 0
    for sample in samples:
        one_feature = sample[0]
        one_label = sample[1]
        nframe_num = one_feature.shape[0]
        nstart = ncur_len
        nend = ncur_len + nframe_num
        bat_feature[nstart:nend, :] = one_feature
        bat_label[nstart:nend, :] = one_label
        ncur_len += nframe_num
    return (bat_feature, bat_label, lod) 
