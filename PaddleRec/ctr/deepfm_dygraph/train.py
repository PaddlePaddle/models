from __future__ import print_function

import os

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

import data_reader
import utility as utils
from network import DeepFM


def train(args):
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
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
            print("start eval model.")
            total_step = 0.0
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

            print("test auc of epoch %d is %.6f" %
                  (epoch, auc_metric_test.eval()))

        #optimizer = fluid.optimizer.SGD(
        #     learning_rate=args.lr,
        #     regularization=fluid.regularizer.L2DecayRegularizer(args.reg))
        optimizer = fluid.optimizer.Adam(
            regularization=fluid.regularizer.L2DecayRegularizer(args.reg))

        # load model if exists
        start_epoch = 0
        if args.checkpoint:
            model_dict, optimizer_dict = fluid.dygraph.load_dygraph(
                args.checkpoint)
            deepfm.set_dict(model_dict)
            optimizer.set_dict(optimizer_dict)
            start_epoch = int(
                os.path.basename(args.checkpoint).split("_")[
                    -1]) + 1  # get next train epoch
            print("load model {} finished.".format(args.checkpoint))

        for epoch in range(start_epoch, args.num_epoch):
            batch_id = 0
            total_loss = 0.0
            auc_metric = fluid.metrics.Auc("ROC")
            print("training epoch {} start.".format(epoch))

            for data in train_reader():
                raw_feat_idx, raw_feat_value, label = zip(*data)
                raw_feat_idx = np.array(raw_feat_idx, dtype=np.int64)
                raw_feat_value = np.array(raw_feat_value, dtype=np.float32)
                label = np.array(label, dtype=np.int64)
                raw_feat_idx, raw_feat_value, label = [
                    to_variable(i)
                    for i in [raw_feat_idx, raw_feat_value, label]
                ]

                predict = deepfm(raw_feat_idx, raw_feat_value, label)

                loss = fluid.layers.log_loss(
                    input=predict,
                    label=fluid.layers.cast(
                        label, dtype="float32"))
                batch_loss = fluid.layers.reduce_sum(loss)

                total_loss += batch_loss.numpy().item()

                batch_loss.backward()
                optimizer.minimize(batch_loss)
                deepfm.clear_gradients()

                # for auc
                predict_2d = fluid.layers.concat([1 - predict, predict], 1)
                auc_metric.update(
                    preds=predict_2d.numpy(), labels=label.numpy())

                if batch_id > 0 and batch_id % 100 == 0:
                    print("epoch: {}, batch_id: {}, loss: {:.6f}, auc: {:.6f}".
                          format(epoch, batch_id,
                                 total_loss / args.batch_size / 100),
                          auc_metric.eval())

                    total_loss = 0.0

                batch_id += 1
            # save model and optimizer
            print("going to save epoch {} model and optimizer.".format(epoch))
            fluid.dygraph.save_dygraph(
                deepfm.state_dict(),
                model_path=os.path.join(args.model_output_dir,
                                        "epoch_" + str(epoch)))
            fluid.dygraph.save_dygraph(
                optimizer.state_dict(),
                model_path=os.path.join(args.model_output_dir,
                                        "epoch_" + str(epoch)))
            print("save epoch {} finished.".format(epoch))
            # eval model
            deepfm.eval()
            eval(epoch)
            deepfm.train()


if __name__ == '__main__':
    args = utils.parse_args()
    utils.print_arguments(args)

    train(args)
