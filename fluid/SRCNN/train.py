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

add_arg('batch_size',        int,   16,         "Batch size.")
add_arg('epoch_num',         int,   15,         "Epoch number.")
add_arg('checkpoint_path',   str,   'model',    "Checkpoint save path.")
add_arg('train_data',        str,   'data/timofte', "Train data path.")
add_arg('val_data',        str,   'data/val_dataset/set5', "Train data path.")
add_arg('img_ext',        str,   'bmp', "image extension.")
add_arg('mode',              str,   'base',     "Model type. one of 'base' or 'custom'")
add_arg('use_gpu',           bool,  False,       "Whether use GPU to train.")

LEARNING_RATE1 = 1e-4
LEARNING_RATE2 = 1e-5
SCALE_FACTOR = 3
# if args.mode == 'custom' modify these
F1=9
F2=3
F3=5
N1=64
N2=32

def net(X, Y, model_struct):        
    # construct net         
    conv1 = fluid.layers.conv2d(X, model_struct.n1, model_struct.f1, act='relu', name='conv1' , 
                                param_attr= fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(scale=0.001),
                                                        name='conv1_w'),
                                bias_attr=fluid.ParamAttr(initializer=fluid.initializer.ConstantInitializer(value=0.),
                                                        name='conv1_b'))
    conv2 = fluid.layers.conv2d(conv1, model_struct.n2, model_struct.f2, act='relu', name='conv2' , 
                                param_attr= fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(scale=0.001),
                                                        name='conv2_w'),
                                bias_attr=fluid.ParamAttr(initializer=fluid.initializer.ConstantInitializer(value=0.),
                                                        name='conv2_b'))
    pred = fluid.layers.conv2d(conv2, 1, model_struct.f3, name='pred', 
                                param_attr= fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(scale=0.001),
                                                        name='pred_w'),
                                bias_attr=fluid.ParamAttr(initializer=fluid.initializer.ConstantInitializer(value=0.),
                                                        name='pred_b'))       
    loss = fluid.layers.reduce_mean(fluid.layers.square(pred - Y))   
    return pred, loss


def train(args):
    if args.mode == 'base':
        model_struct = SRCNNStruct(9, 1, 5, 64, 32)
        output_shape = 21
    elif args.mode == 'custom':
        model_struct = SRCNNStruct(F1, F2, F3, N1, N2)
        output_shape = 33-int(F1/2)*2-int(F2/2)*2-int(F3/2)*2

    X_train = fluid.layers.data(shape=[1, 33, 33], dtype='float32', name='image')
    Y_train = fluid.layers.data(shape=[1, output_shape, output_shape], dtype='float32', name='gdt')
    y_predict, y_loss = net(X_train, Y_train, model_struct)
    Optimizer = fluid.optimizer.AdamOptimizer(learning_rate=LEARNING_RATE1)
    Optimizer_f = fluid.optimizer.AdamOptimizer(learning_rate=LEARNING_RATE2)
    Optimizer.minimize(y_loss, parameter_list=['conv1_w','conv1_b', 'conv2_w', 'conv2_b'])
    Optimizer_f.minimize(y_loss, parameter_list=['pred_w', 'pred_b'])
    
    # read trainSet
    bia_size = int((33-output_shape)/2)
    train_reader = read_data(args.train_data, args.batch_size, args.img_ext, SCALE_FACTOR, bia_size)       
    
    # define place
    if args.use_gpu == True:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    def train_loop(main_program):
        feeder = fluid.DataFeeder(place=place, feed_list=[X_train, Y_train])
        exe.run(fluid.default_startup_program())    
        backprops_cnt=0            
        for epoch in range(args.epoch_num):
            for batch_id, data in enumerate(train_reader()):     
                loss = exe.run(
                    fluid.framework.default_main_program(),
                    feed=feeder.feed(data),
                    fetch_list=[y_loss])             
                if batch_id == 0:   
                    fluid.io.save_inference_model(args.checkpoint_path, ['image'], [y_predict], exe)  
                    val_loss, val_psnr = validation()
                    print("%i\tEpoch: %d \tCur Cost : %f\t Val Cost: %f\t PSNR :%f" % (backprops_cnt, epoch, np.array(loss[0])[0], val_loss, val_psnr))
                backprops_cnt += 1            
        fluid.io.save_inference_model(args.checkpoint_path, ['image'], [y_predict], exe)

    def validation():
        inference_scope = fluid.core.Scope()   
        # use whole img to validate     
      
        for img_name in os.listdir(args.val_data):  
            img_val = cv2.imread(os.path.join(args.val_data, img_name))
            yuv = cv2.cvtColor(img_val, cv2.COLOR_BGR2YCrCb)
            img_y, img_u, img_v = cv2.split(yuv)      
            img_h, img_w = img_y.shape
            img_blur = cv2.GaussianBlur(img_y, (5, 5), 0)
            img_subsample = cv2.resize(img_blur, (int(img_w/SCALE_FACTOR), int(img_h/SCALE_FACTOR)))
            img_input = cv2.resize(img_blur, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
            img_input = np.reshape(img_input, [1,1,img_h, img_w]).astype("float32")    # h,w    
            # cv2.imshow()
            losses = []
            with fluid.scope_guard(inference_scope):
                [inference_program, feed_target_names, fetch_targets] = (
                    fluid.io.load_inference_model(args.checkpoint_path, exe))  
                results = exe.run(inference_program,
                      feed={feed_target_names[0]: img_input},
                      fetch_list=fetch_targets)[0]
                loss = np.mean(np.square(results[0,0]-img_y[bia_size:-bia_size, bia_size:-bia_size]))               
                losses.append(loss) 
        avg_loss = np.sum(np.array(losses))/len(losses)
        psnr = 10 * np.log10(255*255/avg_loss)
        return avg_loss,psnr
    train_loop(fluid.default_main_program())


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    train(args)
