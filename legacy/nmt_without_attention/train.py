import os
import logging
import paddle.v2 as paddle

from network_conf import seq2seq_net

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def train(save_dir_path, source_dict_dim, target_dict_dim):
    '''
    Training function for NMT

    :param save_dir_path: path of the directory to save the trained models.
    :param save_dir_path: str
    :param source_dict_dim: size of source dictionary
    :type source_dict_dim: int
    :param target_dict_dim: size of target dictionary
    :type target_dict_dim: int
    '''
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    # initialize PaddlePaddle
    paddle.init(use_gpu=False, trainer_count=1)

    cost = seq2seq_net(source_dict_dim, target_dict_dim)
    parameters = paddle.parameters.create(cost)

    # define optimization method and the trainer instance
    optimizer = paddle.optimizer.RMSProp(
        learning_rate=1e-3,
        gradient_clipping_threshold=10.0,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4))
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    # define data reader
    wmt14_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(source_dict_dim), buf_size=8192),
        batch_size=8)

    # define the event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if not event.batch_id % 100 and event.batch_id:
                with gzip.open(
                        os.path.join(save_path,
                                     "nmt_without_att_%05d_batch_%05d.tar.gz" %
                                     event.pass_id, event.batch_id), "w") as f:
                    trainer.save_parameter_to_tar(f)

            if event.batch_id and not event.batch_id % 10:
                logger.info("Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))

    # start training
    trainer.train(
        reader=wmt14_reader, event_handler=event_handler, num_passes=2)


if __name__ == '__main__':
    train(save_dir_path="models", source_dict_dim=30000, target_dict_dim=30000)
