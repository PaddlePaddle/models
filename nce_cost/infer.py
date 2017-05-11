# -*- encoding:utf-8 -*-
import numpy as np
import glob
import gzip
import paddle.v2 as paddle
from nce_conf import network_conf


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    prediction_layer = network_conf(
        is_train=False, hidden_size=256, embedding_size=32, dict_size=dict_size)

    models_list = glob.glob('./models/*')
    models_list = sorted(models_list)

    with gzip.open(models_list[-1], 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    idx_word_dict = dict((v, k) for k, v in word_dict.items())
    batch_size = 64
    batch_ins = []
    ins_iter = paddle.dataset.imikolov.test(word_dict, 5)

    infer_data = []
    infer_data_label = []
    for item in paddle.dataset.imikolov.test(word_dict, 5)():
        infer_data.append((item[:4]))
        infer_data_label.append(item[4])
        # Pick up 100 elements of test set. Convenient for show up.
        if len(infer_data_label) == 100:
            break

    feeding = {
        'firstw': 0,
        'secondw': 1,
        'thirdw': 2,
        'fourthw': 3,
        'fifthw': 4
    }

    predictions = paddle.infer(
        output_layer=prediction_layer,
        parameters=parameters,
        input=infer_data,
        feeding=feeding,
        field=['value'])

    for i, (prob, data,
            label) in enumerate(zip(predictions, infer_data, infer_data_label)):
        print '--------------------------'
        print "No.%d Input: " % (i+1) + \
                idx_word_dict[data[0]] + ' ' + \
                idx_word_dict[data[1]] + ' ' + \
                idx_word_dict[data[2]] + ' ' + \
                idx_word_dict[data[3]]
        print 'Ground Truth Output: ' + idx_word_dict[label]
        print 'Predict Output: ' + idx_word_dict[prob.argsort(
            kind='heapsort', axis=0)[-1]]
        print


if __name__ == '__main__':
    main()
