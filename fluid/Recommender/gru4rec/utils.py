import sys
import time
import numpy as np
import paddle.fluid as fluid
import paddle
import data_preprocess as dp
import sort_batch as sortb

def to_lodtensor(data, place):
    """ convert to LODtensor """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

def prepare_data(train_filename, test_filename, batch_size,
		 buffer_size=1000, word_freq_threshold=0, enable_ce=False):
    """ prepare the English Pann Treebank (PTB) data """
    print("start constuct word dict")
    vocab = dp.build_dict(word_freq_threshold,train_filename,test_filename)
    print("construct word dict done\n")
    if enable_ce:
    	train_reader = paddle.batch(
            		dp.train(train_filename,
                	vocab,
                	buffer_size,
                	data_type=dp.DataType.SEQ),
        	batch_size)
    else:
	train_reader = sortb.batch(
                paddle.reader.shuffle(
                        dp.train(train_filename,
                        vocab,
                        buffer_size,
                        data_type=dp.DataType.SEQ),
                buf_size=buffer_size),
                batch_size,batch_size*20)
    test_reader = sortb.batch(
        dp.test(test_filename,
           	vocab, buffer_size, data_type=dp.DataType.SEQ),
       		batch_size,batch_size*20)
    return vocab, train_reader, test_reader
