import numpy as np
import datareader as reader
import paddle.fluid as fluid

class quadrupletloss():
    def __init__(self, 
                 train_batch_size = 80, 
                 test_batch_size = 10,
                 infer_batch_size = 10,
                 samples_each_class = 2,
                 num_gpus=8,
                 margin=0.1):
        self.margin = margin
        self.num_gpus = num_gpus
        self.samples_each_class = samples_each_class
        self.train_batch_size = train_batch_size
        assert(train_batch_size % (samples_each_class*num_gpus) == 0)
        self.class_num = train_batch_size / self.samples_each_class
        self.train_reader = reader.quadruplet_train(train_batch_size, self.class_num, self.samples_each_class)
        self.test_reader = reader.test(test_batch_size)
        self.infer_reader = reader.infer(infer_batch_size)

    def loss(self, input):
        margin = self.margin
        batch_size = self.train_batch_size / self.num_gpus

        fea_dim = input.shape[1] # number of channels
        output = fluid.layers.reshape(input, shape = [-1, fea_dim])

        output = fluid.layers.l2_normalize(output, axis=1)
        #scores = fluid.layers.matmul(output, output, transpose_x=False, transpose_y=True)
        output_t = fluid.layers.transpose(output, perm = [1, 0])
        scores = fluid.layers.mul(x=output, y=output_t)
        mask_np = np.zeros((batch_size, batch_size), dtype=np.float32)
        for i in xrange(batch_size):
            for j in xrange(batch_size):
                if i / self.samples_each_class == j / self.samples_each_class:
                    mask_np[i, j] = 100.
        #mask = fluid.layers.create_tensor(dtype='float32')
        mask = fluid.layers.create_global_var(
                shape=[batch_size, batch_size], value=0, dtype='float32', persistable=True)
        fluid.layers.assign(mask_np, mask)
        scores = fluid.layers.scale(x=scores, scale=-1.0) + mask

        scores_max = fluid.layers.reduce_max(scores, dim=0, keep_dim=True)
        ind_max = fluid.layers.argmax(scores, axis=0)
        ind_max = fluid.layers.cast(x=ind_max, dtype='float32')
        ind2 = fluid.layers.argmax(scores_max, axis=1)
        ind2 = fluid.layers.cast(x=ind2, dtype='int32')
        ind1 = fluid.layers.gather(ind_max, ind2)
        ind1 = fluid.layers.cast(x=ind1, dtype='int32')

        scores_min = fluid.layers.reduce_min(scores, dim=0, keep_dim=True)
        ind_min = fluid.layers.argmin(scores, axis=0)
        ind_min = fluid.layers.cast(x=ind_min, dtype='float32')
        ind4 = fluid.layers.argmin(scores_min, axis=1)
        ind4 = fluid.layers.cast(x=ind4, dtype='int32')
        ind3 = fluid.layers.gather(ind_min, ind4)
        ind3 = fluid.layers.cast(x=ind3, dtype='int32')

        f1 = fluid.layers.gather(output, ind1)
        f2 = fluid.layers.gather(output, ind2)
        f3 = fluid.layers.gather(output, ind3)
        f4 = fluid.layers.gather(output, ind4)

        ind1.stop_gradient = True
        ind2.stop_gradient = True
        ind3.stop_gradient = True
        ind4.stop_gradient = True
        ind_max.stop_gradient = True
        ind_min.stop_gradient = True
        scores_max.stop_gradient = True
        scores_min.stop_gradient = True
        scores.stop_gradient = True
        mask.stop_gradient = True
        output_t.stop_gradient = True
          
        f1_2 = fluid.layers.square(f1 - f2)
        f3_4 = fluid.layers.square(f3 - f4)
        s1 = fluid.layers.reduce_sum(f1_2, dim = 1)
        s2 = fluid.layers.reduce_sum(f3_4, dim = 1)
        s1 = fluid.layers.sqrt(s1)
        s2 = fluid.layers.sqrt(s2)
        loss = fluid.layers.relu(s1 - s2 + margin)
        return loss
