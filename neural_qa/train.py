import sys
import os
import argparse
import numpy as np

import paddle.v2 as paddle

import reader
import utils
import network
import config
from utils import logger


def save_model(trainer, model_save_dir, parameters, pass_id):
    f = os.path.join(model_save_dir, "params_pass_%05d.tar.gz" % pass_id)
    logger.info("model saved to %s" % f)
    with utils.open_file(f, "w") as f:
        trainer.save_parameter_to_tar(f)


def show_parameter_init_info(parameters):
    """
    Print the information of initialization mean and standard deviation of
    parameters

    :param parameters: the parameters created in a model
    """
    logger.info("Parameter init info:")
    for p in parameters:
        p_val = parameters.get(p)
        logger.info(("%-25s : initial_mean=%-7.4f initial_std=%-7.4f "
                     "actual_mean=%-7.4f actual_std=%-7.4f dims=%s") %
                    (p, parameters.__param_conf__[p].initial_mean,
                     parameters.__param_conf__[p].initial_std, p_val.mean(),
                     p_val.std(), parameters.__param_conf__[p].dims))
    logger.info("\n")


def show_parameter_status(parameters):
    """
    Print some statistical information of parameters in a network

    :param parameters: the parameters created in a model
    """
    for p in parameters:
        abs_val = np.abs(parameters.get(p))
        abs_grad = np.abs(parameters.get_grad(p))

        logger.info(
            ("%-25s avg_abs_val=%-10.6f max_val=%-10.6f avg_abs_grad=%-10.6f "
             "max_grad=%-10.6f min_val=%-10.6f min_grad=%-10.6f") %
            (p, abs_val.mean(), abs_val.max(), abs_grad.mean(), abs_grad.max(),
             abs_val.min(), abs_grad.min()))


def train(conf):
    if not os.path.exists(conf.model_save_dir):
        os.makedirs(conf.model_save_dir, mode=0755)

    settings = reader.Settings(
        vocab=conf.vocab,
        is_training=True,
        label_schema=conf.label_schema,
        negative_sample_ratio=conf.negative_sample_ratio,
        hit_ans_negative_sample_ratio=conf.hit_ans_negative_sample_ratio,
        keep_first_b=conf.keep_first_b,
        seed=conf.seed)
    samples_per_pass = conf.batch_size * conf.batches_per_pass
    train_reader = paddle.batch(
        paddle.reader.buffered(
            reader.create_reader(conf.train_data_path, settings,
                                 samples_per_pass),
            size=samples_per_pass),
        batch_size=conf.batch_size)

    # TODO(lipeng17) v2 API does not support parallel_nn yet. Therefore, we can
    # only use CPU currently
    paddle.init(
        use_gpu=conf.use_gpu,
        trainer_count=conf.trainer_count,
        seed=conf.paddle_seed)

    # network config
    cost = network.training_net(conf)

    # create parameters
    # NOTE: parameter values are not initilized here, therefore, we need to
    # print parameter initialization info in the beginning of the first batch
    parameters = paddle.parameters.create(cost)

    # create optimizer
    rmsprop_optimizer = paddle.optimizer.RMSProp(
        learning_rate=conf.learning_rate,
        rho=conf.rho,
        epsilon=conf.epsilon,
        model_average=paddle.optimizer.ModelAverage(
            average_window=conf.average_window,
            max_average_window=conf.max_average_window))

    # create trainer
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=rmsprop_optimizer)

    # begin training network
    def _event_handler(event):
        """
        Define end batch and end pass event handler
        """
        if isinstance(event, paddle.event.EndIteration):
            sys.stderr.write(".")
            batch_num = event.batch_id + 1
            total_batch = conf.batches_per_pass * event.pass_id + batch_num
            if batch_num % conf.log_period == 0:
                sys.stderr.write("\n")
                logger.info("Total batch=%d Batch=%d CurrentCost=%f Eval: %s" \
                        % (total_batch, batch_num, event.cost, event.metrics))

            if batch_num % conf.show_parameter_status_period == 0:
                show_parameter_status(parameters)
        elif isinstance(event, paddle.event.EndPass):
            save_model(trainer, conf.model_save_dir, parameters, event.pass_id)
        elif isinstance(event, paddle.event.BeginIteration):
            if event.batch_id == 0 and event.pass_id == 0:
                show_parameter_init_info(parameters)

    ## for debugging purpose
    #with utils.open_file("config", "w") as config:
    #    print >> config, paddle.layer.parse_network(cost)

    trainer.train(
        reader=train_reader,
        event_handler=_event_handler,
        feeding=network.feeding,
        num_passes=conf.num_passes)

    logger.info("Training has finished.")


def main():
    conf = config.TrainingConfig()

    logger.info("loading word embeddings...")
    conf.vocab, conf.wordvecs = utils.load_wordvecs(conf.word_dict_path,
                                                    conf.wordvecs_path)
    logger.info("loaded")
    logger.info("length of word dictionary is : %d." % len(conf.vocab))

    train(conf)


if __name__ == "__main__":
    main()
