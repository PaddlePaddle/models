from __future__ import print_function

import os

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import logging
import time

import data_reader
import utility as utils
from network import DeepFM

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def infer(args):
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        deepfm = DeepFM(args)

        test_filelist = [
            os.path.join(args.test_data_dir, x)
            for x in os.listdir(args.test_data_dir)
        ]

        test_reader = data_reader.data_reader(
            args.batch_size, test_filelist, args.feat_dict, data_type="test")

        # load model
        if args.checkpoint:
            model_dict, optimizer_dict = fluid.dygraph.load_dygraph(
                args.checkpoint)
            deepfm.set_dict(model_dict)
            logger.info("load model {} finished.".format(args.checkpoint))
        else:
            logger.error("no model to load!")
            logger.error("please set model to load in --checkpoint first.")
            exit(1)

        def eval():
            deepfm.eval()
            logger.info("start eval model.")
            total_step = 0
            batch_begin = time.time()
            auc_metric_test = fluid.metrics.Auc("ROC")
            for data in test_reader():
                total_step += 1
                raw_feat_idx, raw_feat_value, label = zip(*data)
                raw_feat_idx = np.array(raw_feat_idx, dtype=np.int64)
                raw_feat_value = np.array(raw_feat_value, dtype=np.float32)
                label = np.array(label, dtype=np.int64)
                raw_feat_idx, raw_feat_value, label = [
                    to_variable(i)
                    for i in [raw_feat_idx, raw_feat_value, label]
                ]

                predict = deepfm(raw_feat_idx, raw_feat_value, label)

                # for auc
                predict_2d = fluid.layers.concat([1 - predict, predict], 1)
                auc_metric_test.update(
                    preds=predict_2d.numpy(), labels=label.numpy())

                if total_step > 0 and total_step % 100 == 0:
                    logger.info(
                        "TEST --> batch: {} auc: {:.6f} speed: {:.2f} ins/s".
                        format(total_step,
                               auc_metric_test.eval(), 100 * args.batch_size / (
                                   time.time() - batch_begin)))
                    batch_begin = time.time()

            logger.info("test auc is %.6f" % auc_metric_test.eval())

        begin = time.time()
        eval()
        logger.info("test finished, cost %f s" % (time.time() - begin))


if __name__ == '__main__':
    args = utils.parse_args()
    utils.print_arguments(args)

    infer(args)
