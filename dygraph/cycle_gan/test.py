from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import sys
import paddle
import argparse
import functools
import time
import numpy as np
from scipy.misc import imsave
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid import core
import data_reader
from utility import add_arguments, print_arguments, ImagePool
from trainer import *
from paddle.fluid.dygraph.base import to_variable
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)


# yapf: disable
add_arg('batch_size',        int,   1,          "Minibatch size.")
add_arg('epoch',             int,   None,        "The number of weights to be testes.")
add_arg('output',            str,   "./output_0", "The directory the model and the test result to be saved to.")
add_arg('init_model',        str,   './G/',       "The init model file of directory.")

def test():
    with fluid.dygraph.guard():
        A_test_reader = data_reader.a_test_reader()
        B_test_reader = data_reader.b_test_reader()
       
        epoch = args.epoch 
        out_path = args.output + "/eval" + "/" + str(epoch)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cycle_gan = Cycle_Gan("cycle_gan")
        save_dir = args.init_model + str(epoch)
        restore, _ = fluid.dygraph.load_persistables(save_dir)
        cycle_gan.load_dict(restore)
        cycle_gan.eval()
        for data_A , data_B in zip(A_test_reader(), B_test_reader()): 
            A_name = data_A[1]
            B_name = data_B[1]
            print(A_name)
            print(B_name)
            tensor_A = np.array([data_A[0].reshape(3,256,256)]).astype("float32")
            tensor_B = np.array([data_B[0].reshape(3,256,256)]).astype("float32")
            data_A_tmp = to_variable(tensor_A)
            data_B_tmp = to_variable(tensor_B)
            fake_A_temp,fake_B_temp,cyc_A_temp,cyc_B_temp,g_A_loss,g_B_loss,idt_loss_A,idt_loss_B,cyc_A_loss,cyc_B_loss,g_loss = cycle_gan(data_A_tmp,data_B_tmp,True,False,False)
        
            fake_A_temp = np.squeeze(fake_A_temp.numpy()[0]).transpose([1, 2, 0])
            fake_B_temp = np.squeeze(fake_B_temp.numpy()[0]).transpose([1, 2, 0])
            cyc_A_temp = np.squeeze(cyc_A_temp.numpy()[0]).transpose([1, 2, 0])
            cyc_B_temp = np.squeeze(cyc_B_temp.numpy()[0]).transpose([1, 2, 0])
            input_A_temp = np.squeeze(data_A[0]).transpose([1, 2, 0])
            input_B_temp = np.squeeze(data_B[0]).transpose([1, 2, 0])
            imsave(out_path + "/fakeB_" + str(epoch) + "_" + A_name, (
                (fake_B_temp + 1) * 127.5).astype(np.uint8))
            imsave(out_path + "/fakeA_" + str(epoch) + "_" + B_name, (
                (fake_A_temp + 1) * 127.5).astype(np.uint8))
            imsave(out_path + "/cycA_" + str(epoch) + "_" + A_name, (
                (cyc_A_temp + 1) * 127.5).astype(np.uint8))
            imsave(out_path + "/cycB_" + str(epoch) + "_" + B_name, (
                (cyc_B_temp + 1) * 127.5).astype(np.uint8))
            imsave(out_path + "/inputA_" + str(epoch) + "_" + A_name, (
                (input_A_temp + 1) * 127.5).astype(np.uint8))
            imsave(out_path + "/inputB_" + str(epoch) + "_" + B_name, (
                (input_B_temp + 1) * 127.5).astype(np.uint8))

if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    test()
