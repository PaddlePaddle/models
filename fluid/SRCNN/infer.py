from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os

import paddle.fluid as fluid
import numpy as np
import cv2        

from utils import *

# fetch args
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('checkpoint_path',   str,   'model',    "Checkpoint save path.")
add_arg('image_path',        str,   'data/val_dataset/set5/baby_GT.bmp', "Img data path.")
add_arg('show_img',          bool,   False,      "show img or not")
add_arg('only_reconstruct',  bool,   False,      "If True, input image is seemed as subsampled image")
add_arg('scale_factor',      int,   3,      "scale factor")

def reconstruct_img(args):        
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()        
    img_test = cv2.imread(args.image_path)
    yuv_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2YCrCb)       
    img_h, img_w, img_c = img_test.shape
    if args.show_img:
    	cv2.imshow('raw image', img_test)
    
    if args.only_reconstruct == False:
        #   blur image and cubic interpolation
        img_blur = cv2.GaussianBlur(yuv_test.copy(), (5, 5), 0)
        img_subsample = cv2.resize(img_blur, (int(img_w/args.scale_factor), int(img_h/args.scale_factor)))
        img_cubic = cv2.resize(img_blur, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    else:
        img_w *= args.scale_factor
        img_h *= args.scale_factor   
        img_cubic = cv2.resize(yuv_test, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    img_y, img_u, img_v = cv2.split(img_cubic)  
    img_input = np.reshape(img_y, [1,1,img_h, img_w]).astype("float32")    
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = (
            fluid.io.load_inference_model(args.checkpoint_path, exe))  
        results = exe.run(inference_program,
                feed={feed_target_names[0]: img_input},
                fetch_list=fetch_targets)[0]

        result_img = np.array(results)    
        result_img[result_img < 0] = 0
        result_img[result_img >255] = 255
        gap_y = int((img_y.shape[0]-result_img.shape[2])/2)
        gap_x = int((img_y.shape[1]-result_img.shape[3])/2)
        if args.show_img:
            cv2.imshow('input_channel y', img_y)        
        cv2.imwrite(os.path.join(os.path.split(args.image_path)[0],'beforeSR_'+os.path.split(args.image_path)[1]), img_y)
        img_y[gap_y: gap_y + result_img.shape[2],
                gap_x: gap_x + result_img.shape[3]]=result_img
        if args.show_img:
            cv2.imshow('output_channel y', img_y)
            cv2.waitKey(0)

            cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(os.path.split(args.image_path)[0],'afterSR_'+os.path.split(args.image_path)[1]), img_y)
    return img_y
    
if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    reconstruct_img(args)
