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
"""Infer for PWCNet."""
import sys
import pickle
import time
import cv2
import numpy as np
from math import ceil
from scipy.ndimage import imread
from scipy.misc import imsave
import paddle.fluid as fluid
from models.model import PWCDCNet
from src import flow_vis



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


def main():
    im1_fn = 'data/frame_0010.png'
    im2_fn = 'data/frame_0011.png'
    flow_fn = './tmp/frame_0010_pd.flo'
    if len(sys.argv) > 1:
        im1_fn = sys.argv[1]
    if len(sys.argv) > 2:
        im2_fn = sys.argv[2]
    if len(sys.argv) > 3:
        flow_fn = sys.argv[3]

    im_all = [imread(img) for img in [im1_fn, im2_fn]]
    im_all = [im[:, :, :3] for im in im_all]

    # rescale the image size to be multiples of 64
    divisor = 64.
    H = im_all[0].shape[0]
    W = im_all[0].shape[1]
    print('origin shape : ', H, W)

    H_ = int(ceil(H / divisor) * divisor)
    W_ = int(ceil(W / divisor) * divisor)
    print('resize shape: ', H_, W_)
    for i in range(len(im_all)):
        im_all[i] = cv2.resize(im_all[i], (W_, H_))

    for _i, _inputs in enumerate(im_all):
        im_all[_i] = im_all[_i][:, :, ::-1]
        im_all[_i] = 1.0 * im_all[_i] / 255.0
        im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
    im_all = np.concatenate((im_all[0], im_all[1]), axis=0).astype(np.float32)
    im_all = im_all[np.newaxis, :, :, :]

    with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
        im_all = fluid.dygraph.to_variable(im_all)
        im_all, [intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom, intWidth, intHeight] = pad_input(
            im_all)

        model = PWCDCNet("pwcnet")
        model.eval()
        pd_pretrain, _ = fluid.dygraph.load_dygraph("paddle_model/pwc_net_paddle")
        model.set_dict(pd_pretrain)
        start = time.time()
        flo = model(im_all)
        end = time.time()
        print('Time of PWCNet model for one infer step: ', end - start)
        flo = flo[0].numpy() * 20.0
        # scale the flow back to the input size
        flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2)
        flo = flo[intPaddingTop * 2:intPaddingTop * 2 + intHeight * 2,
              intPaddingLeft * 2: intPaddingLeft * 2 + intWidth * 2, :]
        u_ = cv2.resize(flo[:, :, 0], (W, H))
        v_ = cv2.resize(flo[:, :, 1], (W, H))
        u_ *= W / float(W_)
        v_ *= H / float(H_)
        flo = np.dstack((u_, v_))

        # # Apply the coloring (for OpenCV, set convert_to_bgr=True)
        flow_color = flow_vis.flow_to_color(flo, convert_to_bgr=False)
        imsave('./tmp/hsv_pd.png', flow_color)

        writeFlowFile(flow_fn, flo)


if __name__ == '__main__':
    main()


