# encoding=utf-8
from __future__ import print_function
import paddle.v2 as paddle
import sys
import logging

data_dim = 3 * 32 * 32
num_class = 10
paddle.init(use_gpu=False, trainer_count=4)  # 设置不适用GPU，trainer_count=1

logger = logging.getLogger()
formatter = logging.Formatter('%(levelname)-8s:%(message)s')
file_handler = logging.FileHandler('./resNet_paddlepaddle2016.log')  # 日志文件名称
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.info('input=3*32*32')  # 样本图片大小
logger.info('numclass=10')


def bn_conv_layer(input_x, ch_out, filter_size, stride, padding, ch_in=None):
    tmp = paddle.layer.batch_norm(input=input_x, act=paddle.activation.Relu())
    return paddle.layer.img_conv(input=tmp, filter_size=filter_size, num_channels=ch_in, num_filters=ch_out,
                                 stride=stride, padding=padding, act=paddle.activation.Linear(), bias_attr=False)


def shortcut(input_x, ch_in, ch_out, stride=1):
    if ch_in != ch_out:
        return bn_conv_layer(input_x, ch_out, 1, stride, 0)
    else:
        return input_x


def basicblock_changeC(ipt, ch_in, ch_out, stride):
    tmp = bn_conv_layer(ipt, ch_out, 3, stride, 1)
    tmp = bn_conv_layer(tmp, ch_out, 3, 1, 1)
    short = shortcut(ipt, ch_in, ch_out, stride)
    return paddle.layer.addto(input=[tmp, short], act=paddle.activation.Linear())


def layer_warp(block_func, ipt, ch_in, ch_out, count, stride):
    tmp = block_func(ipt, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1)
    return tmp


def resNet_(input_x, layer_num=[3, 4, 6, 3]):
    conv_1 = bn_conv_layer(input_x, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
    basic_1 = layer_warp(basicblock_changeC, conv_1, 16, 16, layer_num[0], 1)
    basic_2 = layer_warp(basicblock_changeC, basic_1, 16, 32, layer_num[1], 2)
    basic_3 = layer_warp(basicblock_changeC, basic_2, 32, 64, layer_num[2], 2)
    basic_4 = layer_warp(basicblock_changeC, basic_3, 64, 128, layer_num[2], 2)
    predict = paddle.layer.img_pool(input=basic_4, pool_size=4, stride=1, pool_type=paddle.pooling.Avg())
    predict_ = paddle.layer.fc(input=predict, size=num_class, act=paddle.activation.Softmax())
    return predict_


input_image = paddle.layer.data(name="image", type=paddle.data_type.dense_vector(data_dim))
images, label = input_image, paddle.layer.data(name="label", type=paddle.data_type.integer_value(num_class))
predict = resNet_(images)
cost = paddle.layer.classification_cost(input=predict, label=label)

parameters = paddle.parameters.create(cost)

optimizer = paddle.optimizer.Momentum(momentum=0.9, regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
                                      learning_rate=0.1 / 128.0, learning_rate_decay_a=0.1,
                                      learning_rate_decay_b=50000 * 100, learning_rate_schedule='discexp')

trainer = paddle.trainer.SGD(cost=cost, parameters=parameters, update_equation=optimizer)

reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=100), batch_size=16)
feeding = {'image': 0, 'label': 1}


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 10 == 0:
            logger.info('pass_id:' + str(event.pass_id) + ',batch_id:' + str(event.batch_id) + ',train_cost:' + str(
                event.cost) + ',s:' + str(event.metrics))
    if isinstance(event, paddle.event.EndPass):
        with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
            trainer.save_parameter_to_tar(f)
        result = trainer.test(
            reader=paddle.batch(
                paddle.dataset.cifar.test10(), batch_size=64),
            feeding=feeding)
        logger.info('pass_id:' + str(event.pass_id) + ',s:' + str(result.metrics))


trainer.train(reader=reader, num_passes=20, event_handler=event_handler, feeding=feeding)

logger.removeHandler(file_handler)
logger.removeHandler(console_handler)
