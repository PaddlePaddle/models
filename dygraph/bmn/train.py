#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.fluid as fluid
import numpy as np
import math
import argparse
import ast
import logging
import sys
import os

from reader import BMNReader
from utils import get_interp1d_mask
from config_utils import *

DATATYPE = 'float32'

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle dynamic graph mode of BMN.")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use data parallel mode to train the model."
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='bmn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='filename to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=9,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='validation epoch interval, 0 for no validation.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default="checkpoint",
        help='path to save train snapshoot')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


# Net
class Conv1D(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters=256,
                 input_size=256,
                 size_k=3,
                 padding=1,
                 groups=1,
                 act="relu",
                 name="conv1d"):
        super(Conv1D, self).__init__(name_scope)
        fan_in = input_size * size_k * 1
        k = 1. / math.sqrt(fan_in)
        param_attr = fluid.initializer.Uniform(low=-k, high=k)
        bias_attr = fluid.initializer.Uniform(low=-k, high=k)

        self._conv2d = fluid.dygraph.Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=(1, size_k),
            stride=1,
            padding=(0, padding),
            groups=groups,
            act=act,
            param_attr=param_attr,
            bias_attr=bias_attr)

    def forward(self, x):
        x = fluid.layers.unsqueeze(input=x, axes=[2])
        x = self._conv2d(x)
        x = fluid.layers.squeeze(input=x, axes=[2])
        return x


class BMN(fluid.dygraph.Layer):
    def __init__(self, name_scope, cfg):
        super(BMN, self).__init__(name_scope)

        #init config
        self.tscale = cfg.MODEL.tscale
        self.dscale = cfg.MODEL.dscale
        self.prop_boundary_ratio = cfg.MODEL.prop_boundary_ratio
        self.num_sample = cfg.MODEL.num_sample
        self.num_sample_perbin = cfg.MODEL.num_sample_perbin

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        # Base Module
        self.b_conv1 = Conv1D(
            name_scope="Base_1",
            input_size=400,
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")
        self.b_conv2 = Conv1D(
            name_scope="Base_2",
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")

        # Temporal Evaluation Module
        self.ts_conv1 = Conv1D(
            name_scope="TEM_s1",
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")
        self.ts_conv2 = Conv1D(
            name_scope="TEM_s2",
            num_filters=1,
            size_k=1,
            padding=0,
            act="sigmoid")
        self.te_conv1 = Conv1D(
            name_scope="TEM_e1",
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")
        self.te_conv2 = Conv1D(
            name_scope="TEM_e2",
            num_filters=1,
            size_k=1,
            padding=0,
            act="sigmoid")

        #Proposal Evaluation Module
        self.p_conv1 = Conv1D(
            name_scope="PEM_1d",
            num_filters=self.hidden_dim_2d,
            size_k=3,
            padding=1,
            act="relu")

        # init to speed up
        sample_mask = get_interp1d_mask(self.tscale, self.dscale,
                                        self.prop_boundary_ratio,
                                        self.num_sample, self.num_sample_perbin)
        self.sample_mask = fluid.dygraph.base.to_variable(sample_mask)
        self.sample_mask.stop_gradient = True

        self.p_conv3d1 = fluid.dygraph.Conv3D(
            name_scope="PEM_3d1",
            num_filters=self.hidden_dim_3d,
            filter_size=(self.num_sample, 1, 1),
            stride=(self.num_sample, 1, 1),
            padding=0,
            act="relu", )

        self.p_conv2d1 = fluid.dygraph.Conv2D(
            name_scope="PEM_2d1",
            num_filters=self.hidden_dim_2d,
            filter_size=1,
            stride=1,
            padding=0,
            act="relu")
        self.p_conv2d2 = fluid.dygraph.Conv2D(
            name_scope="PEM_2d2",
            num_filters=self.hidden_dim_2d,
            filter_size=3,
            stride=1,
            padding=1,
            act="relu")
        self.p_conv2d3 = fluid.dygraph.Conv2D(
            name_scope="PEM_2d3",
            num_filters=self.hidden_dim_2d,
            filter_size=3,
            stride=1,
            padding=1,
            act="relu")
        self.p_conv2d4 = fluid.dygraph.Conv2D(
            name_scope="PEM_2d4",
            num_filters=2,
            filter_size=1,
            stride=1,
            padding=0,
            act="sigmoid")

    def forward(self, x):
        #Base Module
        x = self.b_conv1(x)
        x = self.b_conv2(x)

        #TEM
        xs = self.ts_conv1(x)
        xs = self.ts_conv2(xs)
        xs = fluid.layers.squeeze(xs, axes=[1])
        xe = self.te_conv1(x)
        xe = self.te_conv2(xe)
        xe = fluid.layers.squeeze(xe, axes=[1])

        #PEM
        xp = self.p_conv1(x)
        #BM layer
        xp = fluid.layers.matmul(xp, self.sample_mask)
        xp = fluid.layers.reshape(
            xp, shape=[0, 0, -1, self.dscale, self.tscale])

        xp = self.p_conv3d1(xp)
        xp = fluid.layers.squeeze(xp, axes=[2])
        xp = self.p_conv2d1(xp)
        xp = self.p_conv2d2(xp)
        xp = self.p_conv2d3(xp)
        xp = self.p_conv2d4(xp)
        return xp, xs, xe


def bmn_loss_func(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end,
                  cfg):
    def _get_mask(cfg):
        dscale = cfg.MODEL.dscale
        tscale = cfg.MODEL.tscale
        bm_mask = []
        for idx in range(dscale):
            mask_vector = [1 for i in range(tscale - idx)
                           ] + [0 for i in range(idx)]
            bm_mask.append(mask_vector)
        bm_mask = np.array(bm_mask, dtype=np.float32)
        self_bm_mask = fluid.layers.create_global_var(
            shape=[dscale, tscale], value=0, dtype=DATATYPE, persistable=True)
        fluid.layers.assign(bm_mask, self_bm_mask)
        self_bm_mask.stop_gradient = True
        return self_bm_mask

    def tem_loss_func(pred_start, pred_end, gt_start, gt_end):
        def bi_loss(pred_score, gt_label):
            pred_score = fluid.layers.reshape(
                x=pred_score, shape=[-1], inplace=False)
            gt_label = fluid.layers.reshape(
                x=gt_label, shape=[-1], inplace=False)
            gt_label.stop_gradient = True
            pmask = fluid.layers.cast(x=(gt_label > 0.5), dtype=DATATYPE)
            num_entries = fluid.layers.cast(
                fluid.layers.shape(pmask), dtype=DATATYPE)
            num_positive = fluid.layers.cast(
                fluid.layers.reduce_sum(pmask), dtype=DATATYPE)
            ratio = num_entries / num_positive
            coef_0 = 0.5 * ratio / (ratio - 1)
            coef_1 = 0.5 * ratio
            epsilon = 0.000001
            temp = fluid.layers.log(pred_score + epsilon)
            loss_pos = fluid.layers.elementwise_mul(
                fluid.layers.log(pred_score + epsilon), pmask)
            loss_pos = coef_1 * fluid.layers.reduce_mean(loss_pos)
            loss_neg = fluid.layers.elementwise_mul(
                fluid.layers.log(1.0 - pred_score + epsilon), (1.0 - pmask))
            loss_neg = coef_0 * fluid.layers.reduce_mean(loss_neg)
            loss = -1 * (loss_pos + loss_neg)
            return loss

        loss_start = bi_loss(pred_start, gt_start)
        loss_end = bi_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss

    def pem_reg_loss_func(pred_score, gt_iou_map, mask):

        gt_iou_map = fluid.layers.elementwise_mul(gt_iou_map, mask)

        u_hmask = fluid.layers.cast(x=gt_iou_map > 0.7, dtype=DATATYPE)
        u_mmask = fluid.layers.logical_and(gt_iou_map <= 0.7, gt_iou_map > 0.3)
        u_mmask = fluid.layers.cast(x=u_mmask, dtype=DATATYPE)
        u_lmask = fluid.layers.logical_and(gt_iou_map <= 0.3, gt_iou_map >= 0.)
        u_lmask = fluid.layers.cast(x=u_lmask, dtype=DATATYPE)
        u_lmask = fluid.layers.elementwise_mul(u_lmask, mask)

        num_h = fluid.layers.cast(
            fluid.layers.reduce_sum(u_hmask), dtype=DATATYPE)
        num_m = fluid.layers.cast(
            fluid.layers.reduce_sum(u_mmask), dtype=DATATYPE)
        num_l = fluid.layers.cast(
            fluid.layers.reduce_sum(u_lmask), dtype=DATATYPE)

        r_m = num_h / num_m
        u_smmask = fluid.layers.uniform_random(
            shape=[gt_iou_map.shape[1], gt_iou_map.shape[2]],
            dtype=DATATYPE,
            min=0.0,
            max=1.0)
        u_smmask = fluid.layers.elementwise_mul(u_mmask, u_smmask)
        u_smmask = fluid.layers.cast(x=(u_smmask > (1. - r_m)), dtype=DATATYPE)

        r_l = num_h / num_l
        u_slmask = fluid.layers.uniform_random(
            shape=[gt_iou_map.shape[1], gt_iou_map.shape[2]],
            dtype=DATATYPE,
            min=0.0,
            max=1.0)
        u_slmask = fluid.layers.elementwise_mul(u_lmask, u_slmask)
        u_slmask = fluid.layers.cast(x=(u_slmask > (1. - r_l)), dtype=DATATYPE)

        weights = u_hmask + u_smmask + u_slmask
        weights.stop_gradient = True
        loss = fluid.layers.square_error_cost(pred_score, gt_iou_map)
        loss = fluid.layers.elementwise_mul(loss, weights)
        loss = 0.5 * fluid.layers.reduce_sum(loss) / fluid.layers.reduce_sum(
            weights)

        return loss

    def pem_cls_loss_func(pred_score, gt_iou_map, mask):
        gt_iou_map = fluid.layers.elementwise_mul(gt_iou_map, mask)
        gt_iou_map.stop_gradient = True
        pmask = fluid.layers.cast(x=(gt_iou_map > 0.9), dtype=DATATYPE)
        nmask = fluid.layers.cast(x=(gt_iou_map <= 0.9), dtype=DATATYPE)
        nmask = fluid.layers.elementwise_mul(nmask, mask)

        num_positive = fluid.layers.reduce_sum(pmask)
        num_entries = num_positive + fluid.layers.reduce_sum(nmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = fluid.layers.elementwise_mul(
            fluid.layers.log(pred_score + epsilon), pmask)
        loss_pos = coef_1 * fluid.layers.reduce_sum(loss_pos)
        loss_neg = fluid.layers.elementwise_mul(
            fluid.layers.log(1.0 - pred_score + epsilon), nmask)
        loss_neg = coef_0 * fluid.layers.reduce_sum(loss_neg)
        loss = -1 * (loss_pos + loss_neg) / num_entries
        return loss

    pred_bm_reg = fluid.layers.squeeze(
        fluid.layers.slice(
            pred_bm, axes=[1], starts=[0], ends=[1]), axes=[1])
    pred_bm_cls = fluid.layers.squeeze(
        fluid.layers.slice(
            pred_bm, axes=[1], starts=[1], ends=[2]), axes=[1])

    bm_mask = _get_mask(cfg)

    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)

    tem_loss = tem_loss_func(pred_start, pred_end, gt_start, gt_end)

    loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
    return loss, tem_loss, pem_reg_loss, pem_cls_loss


# Optimizer
def optimizer(cfg):
    bd = [cfg.TRAIN.lr_decay_iter]
    base_lr = cfg.TRAIN.learning_rate
    lr_decay = cfg.TRAIN.learning_rate_decay
    l2_weight_decay = cfg.TRAIN.l2_weight_decay
    lr = [base_lr, base_lr * lr_decay]
    optimizer = fluid.optimizer.Adam(
        fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=l2_weight_decay))
    return optimizer


# Validation
def val_bmn(model, config, args):
    reader = BMNReader(mode="valid", cfg=config)
    val_reader = reader.create_reader()
    for batch_id, data in enumerate(val_reader()):
        video_feat = np.array([item[0] for item in data]).astype(DATATYPE)
        gt_iou_map = np.array([item[1] for item in data]).astype(DATATYPE)
        gt_start = np.array([item[2] for item in data]).astype(DATATYPE)
        gt_end = np.array([item[3] for item in data]).astype(DATATYPE)

        x_data = fluid.dygraph.base.to_variable(video_feat)
        gt_iou_map = fluid.dygraph.base.to_variable(gt_iou_map)
        gt_start = fluid.dygraph.base.to_variable(gt_start)
        gt_end = fluid.dygraph.base.to_variable(gt_end)
        gt_iou_map.stop_gradient = True
        gt_start.stop_gradient = True
        gt_end.stop_gradient = True

        pred_bm, pred_start, pred_end = model(x_data)

        loss, tem_loss, pem_reg_loss, pem_cls_loss = bmn_loss_func(
            pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, config)
        avg_loss = fluid.layers.mean(loss)

        if args.log_interval > 0 and (batch_id % args.log_interval == 0):
            logger.info('[VALID] iter {} '.format(batch_id)
                + '\tLoss = {}, \ttem_loss = {}, \tpem_reg_loss = {}, \tpem_cls_loss = {}'.format(
                '%.04f' % avg_loss.numpy()[0], '%.04f' % tem_loss.numpy()[0], \
                '%.04f' % pem_reg_loss.numpy()[0], '%.04f' % pem_cls_loss.numpy()[0]))


# TRAIN
def train_bmn(args):
    config = parse_config(args.config_file)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))

    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

    with fluid.dygraph.guard(place):
        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
        bmn = BMN("bmn", train_config)
        adam = optimizer(train_config)

        if args.use_data_parallel:
            bmn = fluid.dygraph.parallel.DataParallel(bmn, strategy)

        if args.resume:
            # if resume weights is given, load resume weights directly
            assert os.path.exists(args.resume + ".pdparams"), \
                "Given resume weight dir {} not exist.".format(args.resume)

            model, _ = fluid.dygraph.load_dygraph(args.resume)
            bmn.set_dict(model)

        reader = BMNReader(mode="train", cfg=train_config)
        train_reader = reader.create_reader()
        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        for epoch in range(args.epoch):
            for batch_id, data in enumerate(train_reader()):
                video_feat = np.array(
                    [item[0] for item in data]).astype(DATATYPE)
                gt_iou_map = np.array(
                    [item[1] for item in data]).astype(DATATYPE)
                gt_start = np.array([item[2] for item in data]).astype(DATATYPE)
                gt_end = np.array([item[3] for item in data]).astype(DATATYPE)

                x_data = fluid.dygraph.base.to_variable(video_feat)
                gt_iou_map = fluid.dygraph.base.to_variable(gt_iou_map)
                gt_start = fluid.dygraph.base.to_variable(gt_start)
                gt_end = fluid.dygraph.base.to_variable(gt_end)
                gt_iou_map.stop_gradient = True
                gt_start.stop_gradient = True
                gt_end.stop_gradient = True

                pred_bm, pred_start, pred_end = bmn(x_data)

                loss, tem_loss, pem_reg_loss, pem_cls_loss = bmn_loss_func(
                    pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end,
                    train_config)
                avg_loss = fluid.layers.mean(loss)

                if args.use_data_parallel:
                    avg_loss = bmn.scale_loss(avg_loss)
                    avg_loss.backward()
                    bmn.apply_collective_grads()
                else:
                    avg_loss.backward()

                adam.minimize(avg_loss)

                bmn.clear_gradients()

                if args.log_interval > 0 and (
                        batch_id % args.log_interval == 0):
                    logger.info('[TRAIN] Epoch {}, iter {} '.format(epoch, batch_id)
                         + '\tLoss = {}, \ttem_loss = {}, \tpem_reg_loss = {}, \tpem_cls_loss = {}'.format(
                            '%.04f' % avg_loss.numpy()[0], '%.04f' % tem_loss.numpy()[0], \
                            '%.04f' % pem_reg_loss.numpy()[0], '%.04f' % pem_cls_loss.numpy()[0]))

            logger.info('[TRAIN] Epoch {} training finished'.format(epoch))
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            save_model_name = os.path.join(
                args.save_dir, "bmn_paddle_dy" + "_epoch{}".format(epoch))
            fluid.dygraph.save_dygraph(bmn.state_dict(), save_model_name)

            # validation
            if args.valid_interval > 0 and (epoch + 1
                                            ) % args.valid_interval == 0:
                bmn.eval()
                val_bmn(bmn, valid_config, args)
                bmn.train()

        #save final results
        if fluid.dygraph.parallel.Env().local_rank == 0:
            save_model_name = os.path.join(args.save_dir,
                                           "bmn_paddle_dy" + "_final")
            fluid.dygraph.save_dygraph(bmn.state_dict(), save_model_name)
        logger.info('[TRAIN] training finished')


if __name__ == "__main__":
    args = parse_args()
    train_bmn(args)
