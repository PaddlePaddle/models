import datareader as reader
import paddle.fluid as fluid

class tripletloss():
    def __init__(self, 
                 train_batch_size = 120,
                 test_batch_size = 120, 
                 infer_batch_size = 120,
                 margin=0.1):
        self.train_reader = reader.triplet_train(train_batch_size)
        self.test_reader = reader.test(test_batch_size)
        self.infer_reader = reader.infer(infer_batch_size)
        self.margin = margin

    def loss(self, input):
        margin = self.margin
        fea_dim = input.shape[1] # number of channels
        output = fluid.layers.reshape(input, shape = [-1, 3, fea_dim])
        output = fluid.layers.l2_normalize(output, axis=2)
        anchor, positive, negative = fluid.layers.split(output, num_or_sections = 3, dim = 1)
 
        anchor = fluid.layers.reshape(anchor, shape = [-1, fea_dim])
        positive = fluid.layers.reshape(positive, shape = [-1, fea_dim])
        negative = fluid.layers.reshape(negative, shape = [-1, fea_dim])
 
        a_p = fluid.layers.square(anchor - positive)
        a_n = fluid.layers.square(anchor - negative)
        a_p = fluid.layers.reduce_sum(a_p, dim = 1)
        a_n = fluid.layers.reduce_sum(a_n, dim = 1)
        a_p = fluid.layers.sqrt(a_p)
        a_n = fluid.layers.sqrt(a_n)
        loss = fluid.layers.relu(a_p + margin - a_n)
        return loss
