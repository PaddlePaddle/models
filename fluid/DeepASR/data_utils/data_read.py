"""This model read the sample from disk. 
   use multiprocessing to reading samples
   push samples from one block to multiprocessing queue 
   Todos:
        1. multiprocess read block from disk
"""
import random
import Queue
import numpy as np
import struct
import data_utils.augmentor.trans_mean_variance_norm as trans_mean_variance_norm
import data_utils.augmentor.trans_add_delta as trans_add_delta


class OneBlock(object):
    """ struct for one block :
        contain label, label desc, feature, feature_desc

        Attributes:
            label(str) :  label path of one block
            label_desc(str) : label description path of one block
            feature(str) : feature path of on block
            feature_desc(str) : feature description path of on block
    """

    def __init__(self):
        """the constructor."""

        self.label = "label"
        self.label_desc = "label_desc"
        self.feature = "feature"
        self.feature_desc = "feature_desc"


class DataRead(object):
    """
    Attributes:
        _lblock(obj:`OneBlock`) : the list of OneBlock
        _ndrop_sentence_len(int): dropout the sentence which's frame_num large than _ndrop_sentence_len  
        _que_sample(obj:`Queue`): sample buffer
        _nframe_dim(int): the batch sample frame_dim(todo remove)
        _nstart_block_idx(int): the start block id
        _nload_block_num(int): the block num
    """

    def __init__(self, sfeature_lst, slabel_lst, ndrop_sentence_len=512):
        """
        Args:
            sfeature_lst(str):feature lst path
            slabel_lst(str):label lst path
        Returns:
            None
        """
        self._lblock = []
        self._ndrop_sentence_len = ndrop_sentence_len
        self._que_sample = Queue.Queue()
        self._nframe_dim = 120 * 11
        self._nstart_block_idx = 0
        self._nload_block_num = 1
        self._ndrop_frame_len = 256

        self._load_list(sfeature_lst, slabel_lst)

    def _load_list(self, sfeature_lst, slabel_lst):
        """ load list and shuffle
        Args:
            sfeature_lst(str):feature lst path
            slabel_lst(str):label lst path
        Returns:
            None
        """
        lfeature = open(sfeature_lst).readlines()
        llabel = open(slabel_lst).readlines()
        assert len(llabel) == len(lfeature)
        for i in range(0, len(lfeature), 2):
            one_block = OneBlock()

            one_block.label = llabel[i]
            one_block.label_desc = llabel[i + 1]
            one_block.feature = lfeature[i]
            one_block.feature_desc = lfeature[i + 1]
            self._lblock.append(one_block)

        random.shuffle(self._lblock)

    def _load_one_block(self, lsample, id):
        """read one block by id and push load sample in list lsample 
        Args:
            lsample(list): return sample list
            id(int): block id 
        Returns:
            None
        """
        if id >= len(self._lblock):
            return

        slabel_path = self._lblock[id].label.strip()
        slabel_desc_path = self._lblock[id].label_desc.strip()
        sfeature_path = self._lblock[id].feature.strip()
        sfeature_desc_path = self._lblock[id].feature_desc.strip()

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
            label_data = np.array(label_array, dtype="int64")
            label_data = label_data.reshape((nlabel_frame_num, 1))

            # read feature
            lfeature_split = lfeature_line[i].split()
            nfeature_start = int(lfeature_split[2])
            nfeature_size = int(lfeature_split[3])
            nfeature_frame_num = int(lfeature_split[4])
            nfeature_frame_dim = int(lfeature_split[5])

            file_feature_bin.seek(nfeature_start, 0)
            feature_bytes = file_feature_bin.read(nfeature_size)
            assert nfeature_frame_num * nfeature_frame_dim * 4 == len(
                feature_bytes)
            feature_array = struct.unpack('f' * nfeature_frame_num *
                                          nfeature_frame_dim, feature_bytes)
            feature_data = np.array(feature_array, dtype="float32")
            feature_data = feature_data.reshape(
                (nfeature_frame_num, nfeature_frame_dim))

            #drop long sentence
            if self._ndrop_frame_len < feature_data.shape[0]:
                continue
            lsample.append((feature_data, label_data))

    def get_one_batch(self, nbatch_size):
        """construct one batch(feature, label), batch size is nbatch_size
        Args:
            nbatch_size(int): batch size
        Returns:
            None
        """
        if self._que_sample.empty():
            lsample = self._load_block(
                range(self._nstart_block_idx, self._nstart_block_idx +
                      self._nload_block_num, 1))
            self._move_sample(lsample)
            self._nstart_block_idx += self._nload_block_num

        if self._que_sample.empty():
            self._nstart_block_idx = 0
            return None
        #cal all frame num
        ncur_len = 0
        lod = [0]
        samples = []
        bat_feature = np.zeros((nbatch_size, self._nframe_dim))
        for i in range(nbatch_size):
            # empty clear zero 
            if self._que_sample.empty():
                self._nstart_block_idx = 0
            # copy
            else:
                (one_feature, one_label) = self._que_sample.get()
                samples.append((one_feature, one_label))
                ncur_len += one_feature.shape[0]
                lod.append(ncur_len)

        bat_feature = np.zeros((ncur_len, self._nframe_dim), dtype="float32")
        bat_label = np.zeros((ncur_len, 1), dtype="int64")
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

    def set_trans(self, ltrans):
        """ set transform list
        Args:
            ltrans(list): data tranform list
        Returns:
            None
        """
        self._ltrans = ltrans

    def _load_block(self, lblock_id):
        """read blocks
        """
        lsample = []
        for id in lblock_id:
            self._load_one_block(lsample, id)

        # transform sample
        for (nidx, sample) in enumerate(lsample):
            for trans in self._ltrans:
                sample = trans.perform_trans(sample)
            #print nidx
            lsample[nidx] = sample

        return lsample

    def load_block(self, lblock_id):
        """read blocks
        Args:
            lblock_id(list):the block list id
        Returns:
            None
        """
        lsample = []
        for id in lblock_id:
            self._load_one_block(lsample, id)

        # transform sample
        for (nidx, sample) in enumerate(lsample):
            for trans in self._ltrans:
                sample = trans.perform_trans(sample)
            #print nidx
            lsample[nidx] = sample

        return lsample

    def _move_sample(self, lsample):
        """move sample to queue
        Args:
            lsample(list): one block of samples read from disk
        Returns:
            None
        """
        # random
        random.shuffle(lsample)

        for sample in lsample:
            self._que_sample.put(sample)
