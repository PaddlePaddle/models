import os
import gzip
import numpy as np

import reader
from utils import logger, load_dict, get_embedding
from network_conf import ner_net

import paddle.v2 as paddle
import paddle.v2.evaluator as evaluator


def main(train_data_file,
         test_data_file,
         vocab_file,
         target_file,
         emb_file,
         model_save_dir,
         num_passes=100,
         batch_size=64):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    word_dict = load_dict(vocab_file)
    label_dict = load_dict(target_file)

    word_vector_values = get_embedding(emb_file)

    word_dict_len = len(word_dict)
    label_dict_len = len(label_dict)

    paddle.init(use_gpu=False, trainer_count=1)

    # define network topology
    crf_cost, crf_dec, target = ner_net(word_dict_len, label_dict_len)
    evaluator.sum(name="error", input=crf_dec)
    evaluator.chunk(
        name="ner_chunk",
        input=crf_dec,
        label=target,
        chunk_scheme="IOB",
        num_chunk_types=(label_dict_len - 1) / 2)

    # create parameters
    parameters = paddle.parameters.create(crf_cost)
    parameters.set("emb", word_vector_values)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0,
        learning_rate=2e-4,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        gradient_clipping_threshold=25,
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=10000), )

    trainer = paddle.trainer.SGD(cost=crf_cost,
                                 parameters=parameters,
                                 update_equation=optimizer,
                                 extra_layers=crf_dec)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.data_reader(train_data_file, word_dict, label_dict),
            buf_size=1000),
        batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.data_reader(test_data_file, word_dict, label_dict),
            buf_size=1000),
        batch_size=batch_size)

    feeding = {"word": 0, "mark": 1, "target": 2}

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 5 == 0:
                logger.info("Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))
            if event.batch_id % 50 == 0:
                result = trainer.test(reader=test_reader, feeding=feeding)
                logger.info("\nTest with Pass %d, Batch %d, %s" %
                            (event.pass_id, event.batch_id, result.metrics))

        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with gzip.open(
                    os.path.join(model_save_dir, "params_pass_%d.tar.gz" %
                                 event.pass_id), "w") as f:
                trainer.save_parameter_to_tar(f)

            result = trainer.test(reader=test_reader, feeding=feeding)
            logger.info("\nTest with Pass %d, %s" % (event.pass_id,
                                                     result.metrics))

    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        num_passes=num_passes,
        feeding=feeding)


if __name__ == "__main__":
    main(
        train_data_file="data/train",
        test_data_file="data/test",
        vocab_file="data/vocab.txt",
        target_file="data/target.txt",
        emb_file="data/wordVectors.txt",
        model_save_dir="models/")
