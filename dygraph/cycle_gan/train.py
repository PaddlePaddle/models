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
import six
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   1,          "Minibatch size.")
add_arg('epoch',             int,   20,        "The number of epoched to be trained.")
add_arg('output',            str,   "./output_0", "The directory the model and the test result to be saved to.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('save_checkpoints',  bool,  True,       "Whether to save checkpoints.")
add_arg('run_test',          bool,  True,       "Whether to run test.")
add_arg('use_gpu',           bool,  True,       "Whether to use GPU to train.")
add_arg('profile',           bool,  False,       "Whether to profile.")
add_arg('run_ce',            bool,  False,       "Whether to run for model ce.")
add_arg('changes',           str,   "None",    "The change this time takes.")
# yapf: enable

lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5
tep_per_epoch = 2974

def optimizer_setting():
    lr=0.0002
    optimizer = fluid.optimizer.Adam(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=[
                100 * step_per_epoch, 120 * step_per_epoch,
                140 * step_per_epoch, 160 * step_per_epoch,
                180 * step_per_epoch
            ],
            values=[
                lr , lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
            ]),
        beta1=0.5)    
    return optimizer
def train(args):
    with fluid.dygraph.guard():
        max_images_num = data_reader.max_images_num()
        shuffle = True
        if args.run_ce:
            np.random.seed(10)
            fluid.default_startup_program().random_seed = 90
            max_images_num = 1
            shuffle = False
        data_shape = [-1] + data_reader.image_shape()
        print(data_shape)

    ###    g_A_trainer = G_A(input_A, input_B)
    ###    g_B_trainer = GBTrainer(input_A, input_B)
    ###    gen_trainer = GTrainer(input_A, input_B)
    ###    d_A_trainer = DATrainer(input_B, fake_pool_B)
    ###    d_B_trainer = DBTrainer(input_A, fake_pool_A)

        # prepare environment
        #place = fluid.CPUPlace()
        #if args.use_gpu:
        #    place = fluid.CUDAPlace(0)
        #exe = fluid.Executor(place)2
        #exe.run(fluid.default_startup_program())
        A_pool = ImagePool()
        B_pool = ImagePool()

        A_reader = paddle.batch(
            data_reader.a_reader(shuffle=shuffle), args.batch_size)()
        B_reader = paddle.batch(
            data_reader.b_reader(shuffle=shuffle), args.batch_size)()
        A_test_reader = data_reader.a_test_reader()
        B_test_reader = data_reader.b_test_reader()

	def test(epoch):
            out_path = args.output + "/test"
            if not os.path.exists(out_path):
		os.makedirs(out_path)
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
                #print("!!!!!!!!test()!!!!!!!!",data_B_tmp.numpy())
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

        def checkpoints(epoch):
            out_path = args.output + "/checkpoints/" + str(epoch)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fluid.io.save_persistables(
                exe, out_path + "/gen", main_program=gen_trainer.program)
    ###        fluid.io.save_persistables(
    ###            exe, out_path + "/g_b", main_program=g_B_trainer.program)
            fluid.io.save_persistables(
                exe, out_path + "/d_a", main_program=d_A_trainer.program)
            fluid.io.save_persistables(
                exe, out_path + "/d_b", main_program=d_B_trainer.program)
            print("saved checkpoint to {}".format(out_path))
            sys.stdout.flush()

        cycle_gan = Cycle_Gan("cycle_gan",istrain=True)

        losses = [[], []]
        t_time = 0
        optimizer1 = optimizer_setting()
        optimizer2 = optimizer_setting()
        optimizer3 = optimizer_setting()
##        for param in G.parameters():
##            if param.name[:44]=="g/Cycle_Gan_0/build_generator_resnet_9blocks":
##                print (param.name)
##                vars1.append(var.name)
##
        #cycle_gan.train()
        if args.init_model is not None:
            init_model = args.init_model
            restore = fluid.dygraph.load_persistables(init_model)
            cycle_gan.load_dict(restore)
        for epoch in range(args.epoch):
            batch_id = 0
            cycle_gan.train()
            for i in range(max_images_num):
                #if epoch == 0 and batch_id ==1:
                #    for param in G.parameters():
                #        print(param.name,param.shape)
                data_A = next(A_reader)
                data_B = next(B_reader)
                #print(data_A[0])

                s_time = time.time()
                data_A = np.array([data_A[0].reshape(3,256,256)]).astype("float32")
                data_B = np.array([data_B[0].reshape(3,256,256)]).astype("float32")
                data_A = to_variable(data_A)
                data_B = to_variable(data_B)

                # optimize the g_A network
		fake_A,fake_B,cyc_A,cyc_B,g_A_loss,g_B_loss,idt_loss_A,idt_loss_B,cyc_A_loss,cyc_B_loss,g_loss = cycle_gan(data_A,data_B,True,False,False)

                g_loss_out = g_loss.numpy()

                g_loss.backward()
                #optimizer1.minimize(g_loss)
                vars_G = []
                for param in cycle_gan.parameters():
                    if param.name[:52]=="cycle_gan/Cycle_Gan_0/build_generator_resnet_9blocks":    
			#print (param.name)
                        vars_G.append(param)

                optimizer1.minimize(g_loss,parameter_list=vars_G)                
                #fluid.dygraph.save_persistables(G.state_dict(),"./G")
                cycle_gan.clear_gradients()

                #for param in G.parameters():
                #    dy_param_init_value[param.name] = param.numpy()

                #restore = fluid.dygraph.load_persistables("./G")
                #G.load_dict(restore)


                print("epoch id: %d, batch step: %d, g_loss: %f" % (epoch, batch_id, g_loss_out))



                fake_pool_B = B_pool.pool_image(fake_B).numpy()
                fake_pool_B = np.array([fake_pool_B[0].reshape(3,256,256)]).astype("float32")
                fake_pool_B = to_variable(fake_pool_B)

                fake_pool_A = A_pool.pool_image(fake_A).numpy()
                fake_pool_A = np.array([fake_pool_A[0].reshape(3,256,256)]).astype("float32")
                fake_pool_A = to_variable(fake_pool_A)

                # optimize the d_A network
                rec_B, fake_pool_rec_B = cycle_gan(data_B,fake_pool_B,False,True,False)
                d_loss_A = (fluid.layers.square(fake_pool_rec_B) +
                    fluid.layers.square(rec_B - 1)) / 2.0
                d_loss_A = fluid.layers.reduce_mean(d_loss_A)

                d_loss_A.backward()
                vars_da = []
                for param in cycle_gan.parameters():
                    if param.name[:47]=="cycle_gan/Cycle_Gan_0/build_gen_discriminator_0":
                        #print (param.name)
                        vars_da.append(param)
                optimizer2.minimize(d_loss_A,parameter_list=vars_da)
                #fluid.dygraph.save_persistables(D_A.state_dict(),
                #                                    "./G")
                cycle_gan.clear_gradients()

                #for param in G.parameters():
                #    dy_param_init_value[param.name] = param.numpy()
                #restore = fluid.dygraph.load_persistables("./G")
                #D_A.load_dict(restore)
                #D_A.clear_gradients()

                # optimize the d_B network

                rec_A, fake_pool_rec_A = cycle_gan(data_A,fake_pool_A,False,False,True)
                d_loss_B = (fluid.layers.square(fake_pool_rec_A) +
                    fluid.layers.square(rec_A - 1)) / 2.0
                d_loss_B = fluid.layers.reduce_mean(d_loss_B)

                d_loss_B.backward()
                vars_db = []
                for param in cycle_gan.parameters():
                    if param.name[:47]=="cycle_gan/Cycle_Gan_0/build_gen_discriminator_1":
                        #print (param.name)
                        vars_db.append(param)
                optimizer3.minimize(d_loss_B,parameter_list=vars_db)
                #fluid.dygraph.save_persistables(D_B.state_dict(),
                #                                    "./G")
                cycle_gan.clear_gradients()

                #for param in D_B.parameters():
                #    dy_param_init_value[param.name] = param.numpy()                
                #restore = fluid.dygraph.load_persistables("./G")
                #D_B.load_dict(restore)
                batch_time = time.time() - s_time
                t_time += batch_time
                print(
                    "epoch{}; batch{}; g_loss:{}; d_A_loss: {}; d_B_loss:{} ; \n g_A_loss: {}; g_A_cyc_loss: {}; g_A_idt_loss: {}; g_B_loss: {}; g_B_cyc_loss:  {}; g_B_idt_loss: {};Batch_time_cost: {:.2f}".format(epoch, batch_id,g_loss_out[0],d_loss_A.numpy()[0], d_loss_B.numpy()[0],g_A_loss.numpy()[0],cyc_A_loss.numpy()[0], idt_loss_A.numpy()[0],  g_B_loss.numpy()[0],cyc_B_loss.numpy()[0],idt_loss_B.numpy()[0], batch_time))
                with open('logging_{}.txt'.format(args.changes), 'a') as log_file:
                    now = time.strftime("%c")
                    log_file.write(
                    "time: {}; epoch{}; batch{}; d_A_loss: {}; g_A_loss: {}; \
                    g_A_cyc_loss: {}; g_A_idt_loss: {}; d_B_loss: {}; \
                    g_B_loss: {}; g_B_cyc_loss: {}; g_B_idt_loss: {}; \
                    Batch_time_cost: {:.2f}\n".format(now, epoch, \
                        batch_id, d_loss_A[0], g_A_loss[ 0], cyc_A_loss[0], \
                        idt_loss_A[0], d_loss_B[0], g_A_loss[0], \
                        cyc_B_loss[0], idt_loss_B[0], batch_time))
                losses[0].append(g_A_loss[0])
                losses[1].append(d_loss_A[0])
                sys.stdout.flush()
                batch_id += 1

            #if args.run_test and not args.run_ce:
            #    test(epoch)
            if args.save_checkpoints and not args.run_ce:
                fluid.dygraph.save_persistables(cycle_gan.state_dict(),"./G/{}".format(epoch))
            #test(epoch)

if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    train(args)
