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
"""Trainer for PWCNet."""
import sys
import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.99999"
os.environ["FLAGS_eager_delete_tensor_gb"] = "0"
import pickle
import time
import cv2
import numpy as np
import paddle
import paddle.fluid as fluid
from scipy.misc import imsave
from src import flow_vis
from models.model import PWCDCNet
from data.datasets import FlyingChairs, reader_flyingchairs
from src.multiscaleloss import multiscaleEPE, realEPE
from AverageMeter import *
from my_args import args


def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!");
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def pad_input(x0):
    intWidth = x0.shape[2]
    intHeight = x0.shape[3]
    if intWidth != ((intWidth >> 6) << 6):
        intWidth_pad = (((intWidth >> 6) + 1) << 6)  # more than necessary
        intPaddingLeft = int((intWidth_pad - intWidth) / 2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intWidth_pad = intWidth
        intPaddingLeft = 0
        intPaddingRight = 0

    if intHeight != ((intHeight >> 6) << 6):
        intHeight_pad = (((intHeight >> 6) + 1) << 6)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intHeight_pad = intHeight
        intPaddingTop = 0
        intPaddingBottom = 0

    out = fluid.layers.pad2d(input=x0,
                             paddings=[intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom],
                             mode='edge')

    return out, [intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom, intWidth, intHeight]


def val(model, batch_reader, epoch, batch_num):
    model.eval()
    loss_cnt = AverageMeter()
    for batch_id, data in enumerate(batch_reader()):
        start = time.time()
        im1_data = np.array(
            [x[0] for x in data]).astype('float32')
        im2_data = np.array(
            [x[1] for x in data]).astype('float32')
        flo_data = np.array(
            [x[2] for x in data]).astype('float32')
        step = im1_data.shape[0]

        im_all = np.concatenate((im1_data, im2_data), axis=3).astype(np.float32)
        im_all = im_all / 255.0
        im_all = np.swapaxes(np.swapaxes(im_all, 1, 2), 1, 3)
        label = flo_data / 20.0
        label = np.swapaxes(np.swapaxes(label, 1, 2), 1, 3)

        im_all = fluid.dygraph.to_variable(im_all)
        label = fluid.dygraph.to_variable(label)
        # im_all, [intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom, intWidth, intHeight] = pad_input(
        #     im_all)

        end = time.time()
        read_data_time = end - start
        start = time.time()
        network_output = model(im_all, output_more=False)
        loss = realEPE(network_output, label)
        end = time.time()
        loss_cnt.update(loss.numpy()[0], step)
        print('val epoch {} batch {}/{} run time: {}s read data time {}s loss {}'.format(epoch, batch_id, batch_num,
                                                                                     round(end - start, 2),
                                                                                     round(read_data_time, 2),
                                                                                     loss.numpy()))
    return round(loss_cnt.avg, 4)


def train(model, train_batch_reader, adam, epoch, batch_num, args):
    loss_type = args.loss
    model.train()
    for batch_id, data in enumerate(train_batch_reader()):
        start = time.time()
        im1_data = np.array(
            [x[0] for x in data]).astype('float32')
        im2_data = np.array(
            [x[1] for x in data]).astype('float32')
        flo_data = np.array(
            [x[2] for x in data]).astype('float32')
        im_all = np.concatenate((im1_data, im2_data), axis=3).astype(np.float32)
        im_all = im_all / 255.0
        im_all = np.swapaxes(np.swapaxes(im_all, 1, 2), 1, 3)
        label = flo_data / 20.0
        label = np.swapaxes(np.swapaxes(label, 1, 2), 1, 3)
        if batch_id % 10 == 0:
            im1 = im_all[0, :3, :, :] * 255
            im2 = im_all[0, 3:, :, :] * 255
            im1 = np.swapaxes(np.swapaxes(im1, 0, 1), 1, 2).astype(np.uint8)
            im2 = np.swapaxes(np.swapaxes(im2, 0, 1), 1, 2).astype(np.uint8)

            flo = label[0, :, :, :] * 20
            flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2)
            imsave('./img1.png', im1)
            imsave('./img2.png', im2)
            flow_color = flow_vis.flow_to_color(flo, convert_to_bgr=False)
            imsave('./hsv_pd.png', flow_color)
            H = im_all[0].shape[1]
            W = im_all[0].shape[2]

        im_all = fluid.dygraph.to_variable(im_all)
        label = fluid.dygraph.to_variable(label)
        im_all, [intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom, intWidth, intHeight] = pad_input(
            im_all)

        label, _ = pad_input(label)
        end = time.time()
        read_data_time = end - start
        start = time.time()
        network_output = model(im_all, output_more=True)
        if batch_id % 10 == 0:
            flo = network_output[0][0].numpy() * 20.0
            # scale the flow back to the input size
            flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2)
            flo = flo[intPaddingTop * 2:intPaddingTop * 2 + intHeight * 2,
                                       intPaddingLeft * 2: intPaddingLeft * 2 + intWidth * 2, :]

            u_ = cv2.resize(flo[:, :, 0], (W, H))
            v_ = cv2.resize(flo[:, :, 1], (W, H))
            flo = np.dstack((u_, v_))
            flow_color = flow_vis.flow_to_color(flo, convert_to_bgr=False)
            imsave('./hsv_predict.png', flow_color)
        loss = multiscaleEPE(network_output, label, loss_type, weights=None, sparse=False)

        end = time.time()
        loss.backward()
        if args.use_multi_gpu:
            model.apply_collective_grads()
        adam.minimize(loss)
        model.clear_gradients()
        print('epoch {} batch {}/{} run time: {}s read data time {}s loss {}'.format(epoch, batch_id, batch_num,
                                                                                     round(end - start, 2),
                                                                                     round(read_data_time, 2),
                                                                                     loss.numpy()))


def main():
    print(args)
    if args.use_multi_gpu:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    else:
        place = fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place=place):
        if args.use_multi_gpu:
            strategy = fluid.dygraph.parallel.prepare_context()
        model = PWCDCNet("pwcnet")
        if args.pretrained:
            print('-----------load pretrained model:', args.pretrained)
            pd_pretrain, _ = fluid.dygraph.load_dygraph(args.pretrained)
            model.set_dict(pd_pretrain)

        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.0001, regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0004))
        if args.optimize:
            print('--------------load pretrained model:', args.optimize)
            adam_pretrain, _ = fluid.dygraph.load_dygraph(args.optimize)
            adam.set_dict(adam_pretrain)
        if args.use_multi_gpu:
            model = fluid.dygraph.parallel.DataParallel(model, strategy)

        if args.dataset == 'FlyingChairs':
            train_flyingchairs_dataset = FlyingChairs('train', args, is_cropped=True, txt_file=args.train_val_txt,
                                                      root=args.data_root)
            val_flyingchairs_dataset = FlyingChairs('val', args, is_cropped=False, txt_file=args.train_val_txt,
                                                    root=args.data_root)
        else:
            raise ValueError('dataset name is wrong, please fix it by using args.dataset')

        train_sample_num = len(train_flyingchairs_dataset)
        val_sample_num = len(val_flyingchairs_dataset)
        print('train sample num: ', train_sample_num)
        print('val sample num: ', val_sample_num)
        train_reader = reader_flyingchairs(train_flyingchairs_dataset)
        val_reader = reader_flyingchairs(val_flyingchairs_dataset)
        if args.use_multi_gpu:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)
            val_reader = fluid.contrib.reader.distributed_batch_reader(
                val_reader)
        BATCH_SIZE = args.batch_size
        train_batch_num = round(train_sample_num / BATCH_SIZE)
        val_batch_num = round(val_sample_num / BATCH_SIZE)
        train_batch_reader = paddle.batch(paddle.reader.shuffle(train_reader, buf_size=BATCH_SIZE * 100), BATCH_SIZE,
                                          drop_last=True)
        val_batch_reader = paddle.batch(val_reader, BATCH_SIZE, drop_last=False)
        epoch_num = args.numEpoch
        val_value = 100000000
        rm_best_model = ""

        for epoch in range(epoch_num):
            train(model, train_batch_reader, adam, epoch, train_batch_num, args)
            pd_save_dir = args.model_out_dir
            if not os.path.exists(pd_save_dir):
                os.makedirs(pd_save_dir)
            pd_model_save = os.path.join(pd_save_dir, 'epoch_' + str(epoch) + "_pwc_net_paddle")
            rm_dir = os.path.join(pd_save_dir, 'epoch_' + str(epoch - 1) + "_pwc_net_paddle.pdparams")
            if os.path.exists(rm_dir):
                os.remove(rm_dir)
            if args.use_multi_gpu:
                if fluid.dygraph.parallel.Env().local_rank == 0:
                    fluid.dygraph.save_dygraph(model.state_dict(), pd_model_save)
                    fluid.dygraph.save_dygraph(adam.state_dict(), os.path.join(pd_save_dir, 'adam'))
            else:
                fluid.dygraph.save_dygraph(model.state_dict(), pd_model_save)
                fluid.dygraph.save_dygraph(adam.state_dict(), os.path.join(pd_save_dir, 'adam'))
            val_loss_value = val(model, val_batch_reader, epoch, val_batch_num)
            if val_loss_value < val_value:
                best_model = os.path.join(pd_save_dir, "pwc_net_paddle_" + str(val_loss_value) + '.pdparams')
                os.link(pd_model_save + '.pdparams', best_model)
                if os.path.exists(rm_best_model):
                    os.remove(rm_best_model)
                rm_best_model = best_model
                val_value = val_loss_value


if __name__ == '__main__':
    main()



