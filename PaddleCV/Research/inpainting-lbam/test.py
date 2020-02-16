import os
import sys
import paddle
import paddle.fluid as fluid
import cv2
import numpy as np
import glob
from paddle.fluid.framework import Parameter

from LBAMModel import LBAMModel

import functools
import argparse
from utility import add_arguments, print_arguments
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('imgfn',             str,    None,        "image file name.")
add_arg('maskfn',            str,    None,        "mask file name.")
add_arg('resultfn',           str,    None,         "result file name.")
add_arg('pretrained_model',  str,    None,        "pretrained_model")

def test():
    args = parser.parse_args()
    print_arguments(args)

    pretrained_model = args.pretrained_model

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=pretrained_model, executor=exe, model_filename='model', params_filename='params')

    imgfn  = args.imgfn
    maskfn = args.maskfn
    resultfn = args.resultfn
    if not os.path.exists(args.resultfn):
        os.makedirs(args.resultfn)

    imglist  = sorted(glob.glob(imgfn))
    masklist = sorted(glob.glob(maskfn))

    for imgfn_,maskfn_ in (list(zip(imglist,masklist))):
        print(imgfn_)
        print(maskfn_)
        print('')

        img = cv2.imread(imgfn_)
        mask = cv2.imread(maskfn_)

        img  = img.transpose(2, 0, 1)[::-1]
        img  = img.astype(np.float32)/255.0
        mask = mask.transpose(2, 0, 1)
        mask = mask.astype(np.float32)/255.0

        threshhold = 0.5
        mask = (mask >= threshhold).astype(np.float32)

        # CHW RGB
        mask = 1 - mask
        img = img * mask

        img0 = img
        img  = np.concatenate((img, mask[0:1]), axis=0)

        result = exe.run(inference_program,feed={feed_target_names[0]: img[np.newaxis,:], feed_target_names[1]: mask[np.newaxis,:]}, fetch_list=fetch_targets)

        outimg = result[0][0]
        outimg = outimg * (1-mask) +  img0 * mask

        # BGR HWC
        outimg = outimg[::-1].transpose(1, 2, 0)*255.0


        outfn = os.path.join(args.resultfn, os.path.basename(imgfn_))
        cv2.imwrite(outfn,outimg)


if __name__ == '__main__':
    test()
