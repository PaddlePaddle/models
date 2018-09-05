import data_reader
import os
from paddle.fluid import core
import paddle.fluid as fluid
import paddle
import numpy as np
from network import CycleGAN
from itertools import izip
import random
from scipy.misc import imsave
import sys

class ImagePool(object):

    def __init__(self, pool_size):
        self.pool = []
        self.count = 0
        self.pool_size = pool_size
 
    def pool_image(self, image):
        if self.count < self.pool_size:
            self.pool.append(image)
            self.count += 1
            return image
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self.pool_size-1)
                temp = self.pool[random_id]
                self.pool[random_id] = image
                return temp
            else:
                return image

def train():
    batch_size = 1
    data_shape = [-1, 3, 256, 256]
    input_A = fluid.layers.data(name='input_A', shape=data_shape, dtype='float32')
    input_B = fluid.layers.data(name='input_B', shape=data_shape, dtype='float32')
    fake_pool_A = fluid.layers.data(name='fake_pool_A', shape=data_shape, dtype='float32')
    fake_pool_B = fluid.layers.data(name='fake_pool_B', shape=data_shape, dtype='float32')
    

    optimizer = fluid.optimizer.Adam(learning_rate=0.0002, beta1=0.5)
    gan = CycleGAN(input_A, input_B, fake_pool_A, fake_pool_B, optimizer)

    # prepare environment
    place = fluid.CUDAPlace(0)
#    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program()) 
    A_pool = ImagePool(100)
    B_pool = ImagePool(100)

    a_reader = paddle.batch(data_reader.a_reader(), batch_size)
    b_reader = paddle.batch(data_reader.b_reader(), batch_size)

    def save_training_images(epoch):
        print "Save training images..........."
        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")
        i = 0
        for data_A, data_B in izip(a_reader(), b_reader()):
            tensor_A = core.LoDTensor()
            tensor_B = core.LoDTensor()
            tensor_A.set(data_A, place)
            tensor_B.set(data_B, place)
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = exe.run(gan.infer_program,
                                          fetch_list=[gan.fake_A, gan.fake_B, gan.cyc_A, gan.cyc_B],
                                          feed={"input_A": tensor_A, "input_B": tensor_B})
#            print "fake_A_tmp type: %s; shape: %s" % (type(fake_A_temp), fake_A_temp.shape)
            fake_A_temp = np.squeeze(fake_A_temp[0]).transpose([1,2,0])
            fake_B_temp = np.squeeze(fake_B_temp[0]).transpose([1,2,0])
            cyc_A_temp = np.squeeze(cyc_A_temp[0]).transpose([1,2,0])
            cyc_B_temp = np.squeeze(cyc_B_temp[0]).transpose([1,2,0])
             
            imsave("./output/imgs/fakeB_"+ str(epoch) + "_" + str(i)+".jpg",((fake_A_temp+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/fakeA_"+ str(epoch) + "_" + str(i)+".jpg",((fake_B_temp+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/cycA_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_A_temp+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/cycB_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_B_temp+1)*127.5).astype(np.uint8))
            i += 1
        print "Saved training images"


    for epoch in range(1):
        batch_id = 0
        if epoch % 50 == 1:
            save_training_images(epoch)
        for data_A, data_B in izip(a_reader(), b_reader()):
            batch_id += 1
            tensor_A = core.LoDTensor()
            tensor_B = core.LoDTensor()
            tensor_A.set(data_A, place)
            tensor_B.set(data_B, place)
            # optimize the g_A network
            g_A_loss, fake_B_tmp = exe.run(gan.g_A_program,
                                           fetch_list=[gan.g_loss_A, gan.fake_B],
                                           feed={"input_A": tensor_A, "input_B": tensor_B})
            fake_pool_B = B_pool.pool_image(fake_B_tmp)
             
            # optimize the d_B network
            d_B_loss = exe.run(gan.d_B_program,
                               fetch_list=[gan.d_loss_B],
                               feed={"input_B": tensor_B, "fake_pool_B": fake_pool_B})
        
            # optimize the g_B network
            g_B_loss, fake_A_tmp = exe.run(gan.g_B_program,
                                           fetch_list=[gan.g_loss_B, gan.fake_A],
                                           feed={"input_A": tensor_A, "input_B": tensor_B})
    
            fake_pool_A = A_pool.pool_image(fake_A_tmp)
        
            # optimize the d_A network
            d_A_loss = exe.run(gan.d_A_program,
                               fetch_list=[gan.d_loss_A],
                               feed={"input_A": tensor_A, "fake_pool_A": fake_pool_A})
            print "epoch[%d]; batch[%d]; g_A_loss: %s; d_B_loss: %s; g_B_loss: %s; d_A_loss: %s" % (epoch, batch_id, g_A_loss[0], d_B_loss[0], g_B_loss[0], d_A_loss[0])
            sys.stdout.flush()
 
if __name__ == "__main__":
    train() 
