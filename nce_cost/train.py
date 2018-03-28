import os
import logging
import gzip

import paddle.v2 as paddle
from network_conf import ngram_lm

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def train(model_save_dir):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    paddle.init(use_gpu=False, trainer_count=1)
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    optimizer = paddle.optimizer.Adam(learning_rate=1e-4)

    cost = ngram_lm(hidden_size=128, emb_size=512, dict_size=dict_size)
    parameters = paddle.parameters.create(cost)
    trainer = paddle.trainer.SGD(cost, parameters, optimizer)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id and not event.batch_id % 10:
                logger.info("Pass %d, Batch %d, Cost %f" %
                            (event.pass_id, event.batch_id, event.cost))
        elif isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                paddle.batch(paddle.dataset.imikolov.test(word_dict, 5), 64))
            logger.info("Test Pass %d, Cost %f" % (event.pass_id, result.cost))

            save_path = os.path.join(model_save_dir,
                                     "model_pass_%05d.tar.gz" % event.pass_id)
            logger.info("Save model into %s ..." % save_path)
            with gzip.open(save_path, "w") as f:
                trainer.save_parameter_to_tar(f)

    trainer.train(
        paddle.batch(
            paddle.reader.shuffle(
                lambda: paddle.dataset.imikolov.train(word_dict, 5)(),
                buf_size=1000),
            64),
        num_passes=1000,
        event_handler=event_handler)


if __name__ == "__main__":
    train(model_save_dir="models")
