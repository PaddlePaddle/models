import paddle.v2 as paddle
import data_provider
import vgg_ssd_net
import os, sys
import gzip
import tarfile
from config.pascal_voc_conf import cfg


def train(train_file_list, dev_file_list, data_args, init_model_path):
    optimizer = paddle.optimizer.Momentum(
        momentum=cfg.TRAIN.MOMENTUM,
        learning_rate=cfg.TRAIN.LEARNING_RATE,
        regularization=paddle.optimizer.L2Regularization(
            rate=cfg.TRAIN.L2REGULARIZATION),
        learning_rate_decay_a=cfg.TRAIN.LEARNING_RATE_DECAY_A,
        learning_rate_decay_b=cfg.TRAIN.LEARNING_RATE_DECAY_B,
        learning_rate_schedule=cfg.TRAIN.LEARNING_RATE_SCHEDULE)

    cost, detect_out = vgg_ssd_net.net_conf('train')

    parameters = paddle.parameters.create(cost)
    if not (init_model_path is None):
        assert os.path.isfile(init_model_path), 'Invalid model.'
        parameters.init_from_tar(gzip.open(init_model_path))

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 extra_layers=[detect_out],
                                 update_equation=optimizer)

    feeding = {'image': 0, 'bbox': 1}

    train_reader = paddle.batch(
        data_provider.train(data_args, train_file_list),
        batch_size=cfg.TRAIN.BATCH_SIZE)  # generate a batch image each time

    dev_reader = paddle.batch(
        data_provider.test(data_args, dev_file_list),
        batch_size=cfg.TRAIN.BATCH_SIZE)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1 == 0:
                print "\nPass %d, Batch %d, TrainCost %f, Detection mAP=%f" % \
                        (event.pass_id,
                         event.batch_id,
                         event.cost,
                         event.metrics['detection_evaluator'])
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        if isinstance(event, paddle.event.EndPass):
            with gzip.open('checkpoints/params_pass_%05d.tar.gz' % \
                    event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)
            result = trainer.test(reader=dev_reader, feeding=feeding)
            print "\nTest with Pass %d, TestCost: %f, Detection mAP=%g" % \
                    (event.pass_id,
                     result.cost,
                     result.metrics['detection_evaluator'])

    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        num_passes=cfg.TRAIN.NUM_PASS,
        feeding=feeding)


if __name__ == "__main__":
    paddle.init(use_gpu=True, trainer_count=4)
    data_args = data_provider.Settings(
        data_dir='./data',
        label_file='label_list',
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=[104, 117, 124])
    train(
        train_file_list='./data/trainval.txt',
        dev_file_list='./data/test.txt',
        data_args=data_args,
        init_model_path='./vgg/vgg_model.tar.gz')
