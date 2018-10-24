import argparse
import logging

import numpy as np
import paddle
import paddle.fluid as fluid

import reader
from network_conf import ctr_dnn_model


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle DeepFM example")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="The path of model parameters gz file")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="The path of the dataset to infer")
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help="The size for embedding layer (default:10)")

    return parser.parse_args()


def infer():
    args = parse_args()

    place = fluid.CPUPlace()
    inference_scope = fluid.core.Scope()

    dataset = reader.Dataset()
    test_reader = paddle.batch(dataset.train([args.data_path]), batch_size=1000)

    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    with fluid.framework.program_guard(test_program, startup_program):
        loss, data_list, auc_var, batch_auc_var = ctr_dnn_model(args.embedding_size)

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=data_list, place=place)

    with fluid.scope_guard(inference_scope):
        [inference_program, _, fetch_targets] = fluid.io.load_inference_model(args.model_path, exe)

        def set_zero(var_name):
            param = inference_scope.var(var_name).get_tensor()
            param_array = np.zeros(param._get_dims()).astype("int64")
            param.set(param_array, place)

        auc_states_names = ['_generated_var_2', '_generated_var_3']
        for name in auc_states_names:
            set_zero(name)

        for batch_id, data in enumerate(test_reader()):
            loss_val, auc_val = exe.run(inference_program,
                feed=feeder.feed(data),
                fetch_list=fetch_targets)
            if batch_id % 100 == 0:
                logger.info("TEST --> batch: {} loss: {} auc: {}".format(batch_id, loss_val, auc_val))


if __name__ == '__main__':
    infer()
