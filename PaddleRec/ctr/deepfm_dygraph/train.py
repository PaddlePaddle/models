from __future__ import print_function

import os

import numpy as np
import paddle
import time
import logging

import data_reader
import utility as utils
from network import DeepFM

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args):
    if args.use_gpu:
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CPUPlace()
    paddle.disable_static(place)
    deepfm = DeepFM(args)

    train_filelist = [
        os.path.join(args.train_data_dir, x)
        for x in os.listdir(args.train_data_dir)
    ]
    test_filelist = [
        os.path.join(args.test_data_dir, x)
        for x in os.listdir(args.test_data_dir)
    ]

    train_reader = data_reader.data_reader(
        args.batch_size, train_filelist, args.feat_dict, data_type="train")
    test_reader = data_reader.data_reader(
        args.batch_size, test_filelist, args.feat_dict, data_type="test")

    def eval(epoch):
        deepfm.eval()
        logger.info("start eval model.")
        total_step = 0.0
        auc_metric_test = paddle.metric.Auc("ROC")
        for data in test_reader():
            total_step += 1
            raw_feat_idx, raw_feat_value, label = zip(*data)
            raw_feat_idx = np.array(raw_feat_idx, dtype=np.int64)
            raw_feat_value = np.array(raw_feat_value, dtype=np.float32)
            label = np.array(label, dtype=np.int64)
            raw_feat_idx, raw_feat_value, label = [
                paddle.to_tensor(
                    data=i, dtype=None, place=None, stop_gradient=True)
                for i in [raw_feat_idx, raw_feat_value, label]
            ]

            predict = deepfm(raw_feat_idx, raw_feat_value, label)

            # for auc
            predict_2d = paddle.concat(x=[1 - predict, predict], axis=1)
            auc_metric_test.update(
                preds=predict_2d.numpy(), labels=label.numpy())

        logger.info("test auc of epoch %d is %.6f" %
                    (epoch, auc_metric_test.accumulate()))

    optimizer = paddle.optimizer.Adam(
        parameters=deepfm.parameters(),
        weight_decay=paddle.regularizer.L2Decay(args.reg))

    # load model if exists
    start_epoch = 0
    if args.checkpoint:
        model_dict = paddle.load(os.path.join(args.checkpoint, ".pdparams"))
        optimizer_dict = paddle.load(os.path.join(args.checkpoint, ".pdopt"))
        deepfm.set_dict(model_dict)
        optimizer.set_state_dict(optimizer_dict)
        start_epoch = int(os.path.basename(args.checkpoint).split("_")[
            -1]) + 1  # get next train epoch
        logger.info("load model {} finished.".format(args.checkpoint))

    for epoch in range(start_epoch, args.num_epoch):
        begin = time.time()
        batch_begin = time.time()
        batch_id = 0
        total_loss = 0.0
        auc_metric = paddle.metric.Auc("ROC")
        logger.info("training epoch {} start.".format(epoch))

        for data in train_reader():
            raw_feat_idx, raw_feat_value, label = zip(*data)
            raw_feat_idx = np.array(raw_feat_idx, dtype=np.int64)
            raw_feat_value = np.array(raw_feat_value, dtype=np.float32)
            label = np.array(label, dtype=np.int64)
            raw_feat_idx, raw_feat_value, label = [
                paddle.to_tensor(
                    data=i, dtype=None, place=None, stop_gradient=True)
                for i in [raw_feat_idx, raw_feat_value, label]
            ]

            predict = deepfm(raw_feat_idx, raw_feat_value, label)

            loss = paddle.nn.functional.log_loss(
                input=predict, label=paddle.cast(
                    label, dtype="float32"))
            batch_loss = paddle.sum(loss)

            total_loss += batch_loss.numpy().item()

            batch_loss.backward()
            optimizer.minimize(batch_loss)
            deepfm.clear_gradients()

            # for auc
            predict_2d = paddle.concat(x=[1 - predict, predict], axis=1)
            auc_metric.update(preds=predict_2d.numpy(), labels=label.numpy())

            if batch_id > 0 and batch_id % 100 == 0:
                logger.info(
                    "epoch: {}, batch_id: {}, loss: {:.6f}, auc: {:.6f}, speed: {:.2f} ins/s".
                    format(epoch, batch_id, total_loss / args.batch_size / 100,
                           auc_metric.accumulate(), 100 * args.batch_size / (
                               time.time() - batch_begin)))
                batch_begin = time.time()
                total_loss = 0.0

            batch_id += 1
        logger.info("epoch %d is finished and takes %f s" %
                    (epoch, time.time() - begin))
        # save model and optimizer
        logger.info("going to save epoch {} model and optimizer.".format(epoch))
        paddle.save(
            deepfm.state_dict(),
            path=os.path.join(args.model_output_dir, "epoch_" + str(epoch),
                              ".pdparams"))
        paddle.save(
            optimizer.state_dict(),
            path=os.path.join(args.model_output_dir, "epoch_" + str(epoch),
                              ".pdopt"))
        logger.info("save epoch {} finished.".format(epoch))
        # eval model
        deepfm.eval()
        eval(epoch)
        deepfm.train()
    paddle.enable_static()


if __name__ == '__main__':
    args = utils.parse_args()
    utils.print_arguments(args)

    train(args)
