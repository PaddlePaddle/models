#!/usr/bin/env python
#coding:gbk
from grid_lstm_net import *


def generate():
    gen_creator = paddle.dataset.wmt14.gen(source_language_dict_dim)
    gen_data = []
    for item in gen_creator():
        gen_data.append((item[0], ))

    beam_gen = grid_lstm_net(source_language_dict_dim, target_language_dict_dim,
                             True)
    #get model
    parameters = paddle.dataset.wmt14.model()
    beam_result = paddle.infer(
        output_layer=beam_gen,
        parameters=parameters,
        input=gen_data,
        field=['prob', 'id'])

    #get th dictionary
    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(source_language_dict_dim)

    seq_list = []
    seq = []
    for w in beam_result[1]:
        if w != -1:
            seq.append(w)
        else:
            seq_list.append(' '.join([trg_dict.get(w) for w in seq[1:]]))
            seq = []

    prob = beam_result[0]
    for i in xrange(len(gen_data)):
        print "\n*******************************"
        print "sec:", ' '.join([src_dict.get(w) for w in gen_data[i][0]]), "\n"
        for j in xrange(beam_size):
            print "prob = %f:" % (prob[i][j]), seq_list[i * beam_size + j]


if __name__ == '__main__':
    generate()
