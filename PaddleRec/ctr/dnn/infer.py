# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os
import time
import six
import numpy as np
import logging
import argparse
import paddle
import paddle.fluid as fluid
from network_conf import CTR
import feed_generator as generator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle CTR-DNN example")
    # -------------Data & Model Path-------------
    parser.add_argument(
        '--test_files_path',
        type=str,
        default='./test_data',
        help="The path of testing dataset")
    parser.add_argument(
        '--model_path',
        type=str,
        default='models',
        help='The path for model to store (default: models)')

    # -------------Running parameter-------------
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")
    parser.add_argument(
        '--infer_epoch',
        type=int,
        default=0,
        help='Specify which epoch to run infer'
    )
    # -------------Network parameter-------------
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help="The size for embedding layer (default:10)")
    parser.add_argument(
        '--sparse_feature_dim',
        type=int,
        default=1000001,
        help='sparse feature hashing space for index processing')
    parser.add_argument(
        '--dense_feature_dim',
        type=int,
        default=13,
        help='dense feature shape')

    # -------------device parameter-------------
    parser.add_argument(
        '--is_local',
        type=int,
        default=0,
        help='Local train or distributed train (default: 1)')
    parser.add_argument(
        '--is_cloud',
        type=int,
        default=0,
        help='Local train or distributed train on paddlecloud (default: 0)')

    return parser.parse_args()


def print_arguments(args):
    """
    print arguments
    """
    logger.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        logger.info('%s: %s' % (arg, value))
    logger.info('------------------------------------------------')


def run_infer(args, model_path):
    place = fluid.CPUPlace()
    train_generator = generator.CriteoDataset(args.sparse_feature_dim)
    file_list = [
        os.path.join(args.test_files_path, x) for x in os.listdir(args.test_files_path)
    ]
    test_reader = paddle.batch(train_generator.test(file_list),
                               batch_size=args.batch_size)
    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    ctr_model = CTR()

    def set_zero():
        auc_states_names = [
            '_generated_var_0', '_generated_var_1', '_generated_var_2',
            '_generated_var_3'
        ]
        for name in auc_states_names:
            param = fluid.global_scope().var(name).get_tensor()
            if param:
                param_array = np.zeros(param._get_dims()).astype("int64")
                param.set(param_array, place)

    with fluid.framework.program_guard(test_program, startup_program):
        with fluid.unique_name.guard():
            inputs = ctr_model.input_data(args)
            loss, auc_var = ctr_model.net(inputs, args)

            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(feed_list=inputs, place=place)

            if args.is_cloud:
                fluid.io.load_persistables(
                    executor=exe,
                    dirname=model_path,
                    main_program=fluid.default_main_program())
            elif args.is_local:
                fluid.load(fluid.default_main_program(),
                           os.path.join(model_path, "checkpoint"), exe)
            set_zero()

            run_index = 0
            infer_auc = 0
            L = []
            for batch_id, data in enumerate(test_reader()):
                loss_val, auc_val = exe.run(test_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[loss, auc_var])
                run_index += 1
                infer_auc = auc_val
                L.append(loss_val / args.batch_size)
                if batch_id % 100 == 0:
                    logger.info("TEST --> batch: {} loss: {} auc: {}".format(
                        batch_id, loss_val / args.batch_size, auc_val))

            infer_loss = np.mean(L)
            infer_result = {}
            infer_result['loss'] = infer_loss
            infer_result['auc'] = infer_auc
            log_path = os.path.join(model_path, 'infer_result.log')
            logger.info(str(infer_result))
            with open(log_path, 'w+') as f:
                f.write(str(infer_result))
            logger.info("Inference complete")
    return infer_result


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    model_list = []
    for _, dir, _ in os.walk(args.model_path):
        for model in dir:
            if "epoch" in model and args.infer_epoch == int(model.split('_')[-1]):
                path = os.path.join(args.model_path, model)
                model_list.append(path)

    if len(model_list) == 0:
        logger.info("There is no satisfactory model {} at path {}, please check your start command & env. ".format(
            str("epoch_")+str(args.infer_epoch), args.model_path))

    for model in model_list:
        logger.info("Test model {}".format(model))
        run_infer(args, model)
