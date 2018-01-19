import os
import logging
import gzip

import paddle.v2 as paddle
from network_conf import ngram_lm

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def main(save_dir="models"):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    paddle.init(use_gpu=False, trainer_count=1)
    word_dict = paddle.dataset.imikolov.build_dict(min_word_freq=2)
    dict_size = len(word_dict)

    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=3e-3,
        regularization=paddle.optimizer.L2Regularization(8e-4))

    cost = ngram_lm(hidden_size=256, embed_size=32, dict_size=dict_size)

    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=3e-3,
        regularization=paddle.optimizer.L2Regularization(8e-4))
    trainer = paddle.trainer.SGD(cost, parameters, adam_optimizer)

    def event_handler(event):
        if isinstance(event, paddle.event.EndPass):
            model_name = os.path.join(save_dir, "hsigmoid_pass_%05d.tar.gz" %
                                      event.pass_id)
            logger.info("Save model into %s ..." % model_name)
            with gzip.open(model_name, "w") as f:
                trainer.save_parameter_to_tar(f)

        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id and event.batch_id % 10 == 0:
                result = trainer.test(
                    paddle.batch(
                        paddle.dataset.imikolov.test(word_dict, 5), 32))
                logger.info(
                    "Pass %d, Batch %d, Cost %f, Test Cost %f" %
                    (event.pass_id, event.batch_id, event.cost, result.cost))

    trainer.train(
        paddle.batch(
            paddle.reader.shuffle(
                lambda: paddle.dataset.imikolov.train(word_dict, 5)(),
                buf_size=1000),
            64),
        num_passes=30,
        event_handler=event_handler)


if __name__ == "__main__":
    main()
