import argparse
import logging

import numpy as np
# disable gpu training for this example 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import paddle
import paddle.fluid as fluid

import reader
from network_conf import ctr_dnn_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
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
    parser.add_argument(
        '--sparse_feature_dim',
        type=int,
        default=1000001,
        help="The size for embedding layer (default:1000001)")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")

    return parser.parse_args()


def infer():
    args = parse_args()

    place = fluid.CPUPlace()
    inference_scope = fluid.Scope()

    dataset = reader.CriteoDataset(args.sparse_feature_dim)
    test_reader = paddle.batch(
        dataset.test([args.data_path]), batch_size=args.batch_size)

    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    with fluid.framework.program_guard(test_program, startup_program):
        loss, auc_var, batch_auc_var, _, data_list = ctr_dnn_model(
            args.embedding_size, args.sparse_feature_dim, False)

        exe = fluid.Executor(place)

        feeder = fluid.DataFeeder(feed_list=data_list, place=place)

        fluid.io.load_persistables(
            executor=exe,
            dirname=args.model_path,
            main_program=fluid.default_main_program())

        def set_zero(var_name):
            param = inference_scope.var(var_name).get_tensor()
            param_array = np.zeros(param._get_dims()).astype("int64")
            param.set(param_array, place)

        auc_states_names = ['_generated_var_2', '_generated_var_3']
        for name in auc_states_names:
            set_zero(name)

        for batch_id, data in enumerate(test_reader()):
            loss_val, auc_val = exe.run(test_program,
                                        feed=feeder.feed(data),
                                        fetch_list=[loss, auc_var])
            if batch_id % 100 == 0:
                logger.info("TEST --> batch: {} loss: {} auc: {}".format(
                    batch_id, loss_val / args.batch_size, auc_val))


if __name__ == '__main__':
    infer()
