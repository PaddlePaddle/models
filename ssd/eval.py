import paddle.v2 as paddle
import data_provider
import vgg_ssd_net
import os, sys
import gzip
from config.pascal_voc_conf import cfg


def eval(eval_file_list, batch_size, data_args, model_path):
    cost, detect_out = vgg_ssd_net.net_conf(mode='eval')

    assert os.path.isfile(model_path), 'Invalid model.'
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))

    optimizer = paddle.optimizer.Momentum()

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 extra_layers=[detect_out],
                                 update_equation=optimizer)

    feeding = {'image': 0, 'bbox': 1}

    reader = paddle.batch(
        data_provider.test(data_args, eval_file_list), batch_size=batch_size)

    result = trainer.test(reader=reader, feeding=feeding)

    print "TestCost: %f, Detection mAP=%g" % \
            (result.cost, result.metrics['detection_evaluator'])


if __name__ == "__main__":
    paddle.init(use_gpu=True, trainer_count=4)  # use 4 gpus

    data_args = data_provider.Settings(
        data_dir='./data',
        label_file='label_list',
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=[104, 117, 124])

    eval(
        eval_file_list='./data/test.txt',
        batch_size=4,
        data_args=data_args,
        model_path='models/pass-00000.tar.gz')
