# -*- coding:utf-8 -*-
import sys, os
import numpy as np
import paddle.v2 as paddle
from attentive_reader import attentive_reader_net
from impatient_reader import impatient_reader_net
from two_layer_lstm import two_layer_lstm_net

with_gpu = os.getenv('WITH_GPU', '0') != '0'


def save_model(trainer, parameters, save_path):
    with open(save_path, 'w') as f:
        trainer.save_parameter_to_tar(f)

def main():
    paddle.init(use_gpu=True, trainer_count=1)
    is_generating = True

    # 定义dict的维度
    dict_size = 30000
    source_dict_dim = target_dict_dim = dict_size

    # 训练网络
    if not is_generating:
        # 定义方法并优化训练器
        optimizer = paddle.optimizer.Adam(
            learning_rate=5e-5,
            regularization=paddle.optimizer.L2Regularization(rate=8e-4))

        cost = two_layer_lstm_net(source_dict_dim, target_dict_dim, is_generating)
        parameters = paddle.parameters.create(cost)

        trainer = paddle.trainer.SGD(
            cost=cost, parameters=parameters, update_equation=optimizer)
        # 设置数据集
        wmt14_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.wmt14.train(dict_size), buf_size=8192),
            batch_size=4)

        # 设置event_handler
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 10 == 0:
                    print("\nPass %d, Batch %d, Cost %f, %s" %
                          (event.pass_id, event.batch_id, event.cost,
                           event.metrics))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                if not event.batch_id % 10:
                    save_path = 'params_pass_%05d_batch_%05d.tar' % (
                        event.pass_id, event.batch_id)
                    save_model(trainer, parameters, save_path)

            if isinstance(event, paddle.event.EndPass):
                # save parameters
                save_path = 'params_pass_%05d.tar' % (event.pass_id)
                save_model(trainer, parameters, save_path)

        # 开始训练
        trainer.train(
            reader=wmt14_reader, event_handler=event_handler, num_passes=2)

    # 生成问句并寻找回答
    else:
        # 只针对头三个问句
        gen_data = []
        gen_num = 3
        for item in paddle.dataset.wmt14.gen(dict_size)():
            gen_data.append([item[0]])
            if len(gen_data) == gen_num:
                break

        beam_size = 3
        beam_gen = two_layer_lstm_net(source_dict_dim, target_dict_dim,
                                  is_generating, beam_size)

        # 设置训练好的模型 bleu = 26.92
        parameters = paddle.dataset.wmt14.model()

        # prob 是回答准确的概率
        beam_result = paddle.infer(
            output_layer=beam_gen,
            parameters=parameters,
            input=gen_data,
            field=['prob', 'id'])

        # 载入数据集
        src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)

        gen_sen_idx = np.where(beam_result[1] == -1)[0]
        assert len(gen_sen_idx) == len(gen_data) * beam_size

        # 生成回答
        start_pos, end_pos = 1, 0
        for i, sample in enumerate(gen_data):
            print(
                " ".join([src_dict[w] for w in sample[0][1:-1]])
            )
            for j in xrange(beam_size):
                end_pos = gen_sen_idx[i * beam_size + j]
                print("%.4f\t%s" % (beam_result[0][i][j], " ".join(
                    trg_dict[w] for w in beam_result[1][start_pos:end_pos])))
                start_pos = end_pos + 2
            print("\n")


if __name__ == '__main__':
    main()
