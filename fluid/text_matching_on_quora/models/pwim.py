import paddle.fluid as fluid
import numpy

class pwimNet():
    """pwim net"""

    def __init__(self, config):
         self._config = config

    def __call__(self, seq1, seq2, label):
        return self.body(seq1, seq2, label, self._config)

    def body(self, seq1, seq2, label, config):
        """Body function"""
        def context_modeling(seq, name):
            #if config.share_weight_btw_seq == Ture:
            #    name = 'seq'

            seq_embedding = fluid.layers.embedding(input=seq, 
                                                   size=[config.dict_dim, config.emb_dim],
                                                   param_attr='emb.w') 

            fw = fluid.layers.fc(input=seq_embedding, 
                                 size=config.fc_dim * 4,
                                 param_attr=name+'.fw_fc.w',
                                 bias_attr=False)
            h, _ = fluid.layers.dynamic_lstm(input=fw,
                                             size=config.fc_dim * 4,
                                             is_reverse=False,
                                             param_attr=name+'.lstm_w',
                                             bias_attr=name+'.lstm_b')

            rv = fluid.layers.fc(input=seq_embedding, 
                                 size=config.fc_dim * 4,  
                                 param_attr=name+'.rv_fc.w', 
                                 bias_attr=False)
            r_h, _ = fluid.layers.dynamic_lstm(input=rv,
                                               size=config.fc_dim * 4,
                                               is_reverse=True,
                                               param_attr=name+'.reversed_lstm_w',
                                               bias_attr=name+'.reversed_lstm_b')
            
            # LoDTensor turn to tensor by padding
            pad_value = fluid.layers.assign(input=numpy.array([0], dtype=numpy.float32))
            
            # it can be 32 or 48 in the paper
            padding_len = config.seq_limit_len # defalut = 48
            h_tensor, mask_f = fluid.layers.sequence_pad(x=h, pad_value=pad_value, maxlen=padding_len)
            r_h_tensor, mask_b = fluid.layers.sequence_pad(x=r_h, pad_value=pad_value, maxlen=padding_len)
            return (h_tensor, r_h_tensor) 

        def pairwise_word_interaction_modeling(h_f0, h_b0, h_f1, h_b1):
            def l2euclid_layer(x, y):
                xy = fluid.layers.matmul(x, y, transpose_y=True)
                xy = fluid.layers.scale(xy, scale=2)
                
                square_x = fluid.layers.square(x)
                sum2_x = fluid.layers.reduce_sum(square_x, dim=2)
                res = fluid.layers.elementwise_add(xy, sum2_x, axis=0) # => broadcast
                
                square_y = fluid.layers.square(y)
                sum2_y = fluid.layers.reduce_sum(square_y, dim=2)
                # transpose for broadcast
                res = fluid.layers.transpose(res, perm=[0, 2, 1])
                res = fluid.layers.elementwise_add(res, sum2_y, axis=0) # => broadcast
                res = fluid.layers.transpose(res, perm=[0, 2, 1])
                return fluid.layers.sqrt(res)
            
            def cos_layer(x, y):
                mul = fluid.layers.matmul(x, y, transpose_y=True)
                normalize_x = fluid.layers.square(x)
                normalize_x = fluid.layers.reduce_sum(normalize_x, dim=2)
                normalize_x = fluid.layers.sqrt(normalize_x)
                normalize_y = fluid.layers.square(y)
                normalize_y = fluid.layers.reduce_sum(normalize_y, dim=2)
                normalize_y = fluid.layers.sqrt(normalize_y)
                normalize_x = fluid.layers.unsqueeze(normalize_x, axes=[2])
                normalize_y = fluid.layers.unsqueeze(normalize_y, axes=[2])
                normalize = fluid.layers.matmul(normalize_x, normalize_y, transpose_y=True)
                # element in normalize can not be zero
                normalize = fluid.layers.elementwise_add(normalize,
                        fluid.layers.fill_constant_batch_size_like(input=x, 
                                                                   shape=[-1, config.seq_limit_len, config.seq_limit_len],
                                                                   value=1e-12,
                                                                   dtype="float32"))
                return fluid.layers.elementwise_div(mul, normalize)
            
            #TODO: mask
            h_bi0 = fluid.layers.concat(input=[h_f0, h_b0], axis=2)
            h_bi1 = fluid.layers.concat(input=[h_f1, h_b1], axis=2)
            h_add0 = fluid.layers.elementwise_add(h_f0, h_b0)
            h_add1 = fluid.layers.elementwise_add(h_f1, h_b1)
            
            # cos
            simCube1 = fluid.layers.unsqueeze(cos_layer(h_bi0, h_bi1), axes=[1])
            simCube4 = fluid.layers.unsqueeze(cos_layer(h_f0, h_f1), axes=[1])
            simCube7 = fluid.layers.unsqueeze(cos_layer(h_b0, h_b1), axes=[1])
            simCube10 = fluid.layers.unsqueeze(cos_layer(h_add0, h_add1), axes=[1])
            
            # L2-Euclid
            simCube2 = fluid.layers.unsqueeze(l2euclid_layer(h_bi0, h_bi1), axes=[1])
            simCube5 = fluid.layers.unsqueeze(l2euclid_layer(h_f0, h_f1), axes=[1])
            simCube8 = fluid.layers.unsqueeze(l2euclid_layer(h_b0, h_b1), axes=[1])
            simCube11 = fluid.layers.unsqueeze(l2euclid_layer(h_add0, h_add1), axes=[1])
            # DotProduct
            simCube3 = fluid.layers.unsqueeze(fluid.layers.matmul(h_bi0, h_bi1, transpose_y=True), axes=[1])
            simCube6 = fluid.layers.unsqueeze(fluid.layers.matmul(h_f0, h_f1, transpose_y=True), axes=[1])
            simCube9 = fluid.layers.unsqueeze(fluid.layers.matmul(h_b0, h_b1, transpose_y=True), axes=[1])
            simCube12 = fluid.layers.unsqueeze(fluid.layers.matmul(h_add0, h_add1, transpose_y=True), axes=[1])
            
            simCube = fluid.layers.concat(input=[simCube1, simCube2, simCube3, 
                                                 simCube4, simCube5, simCube6,
                                                 simCube7, simCube8, simCube9,
                                                 simCube10, simCube11, simCube12], axis=1)
            return simCube

        def similarity_focus_layer(simCube):
            # zero padding may have some effect on the results, but it should be small
            mask = fluid.layers.similarity_focus(simCube, 1, [9, 10])

            mask = fluid.layers.elementwise_add(
                        fluid.layers.scale(mask, scale=0.9), 
                        fluid.layers.fill_constant_batch_size_like(input=mask,
                                                                   shape=[-1, 12, config.seq_limit_len, config.seq_limit_len],
                                                                   value=0.1,
                                                                   dtype="float32"))
            
            focusCube = fluid.layers.elementwise_mul(simCube, mask)
            
            simCube13 = fluid.layers.fill_constant_batch_size_like(input=focusCube, 
                                                                   shape=[-1, 1, config.seq_limit_len, config.seq_limit_len],
                                                                   value=1,
                                                                   dtype="float32")
            focusCube = fluid.layers.concat(input=[focusCube, simCube13], axis=1)
            return focusCube

        def conv_relu_pool(input, 
                conv_num_filters, conv_filter_size, conv_stride, conv_padding,
                pool_size, pool_type, pool_stride, pool_padding=0):
            relu = fluid.layers.conv2d(input=input,
                                       num_filters=conv_num_filters,
                                       filter_size=conv_filter_size,
                                       stride=conv_stride,
                                       padding=conv_padding,
                                       act='relu')
            pool = fluid.layers.pool2d(input=relu,
                                       pool_size=pool_size,
                                       pool_type=pool_type,
                                       pool_padding=pool_padding,
                                       pool_stride=pool_stride)
            return pool


        def similarity_classification_with_deep_convolutional_neural_networks(focusCube):
            t1 = conv_relu_pool(input=focusCube,
                                conv_num_filters=128,
                                conv_filter_size=3,
                                conv_stride=1,
                                conv_padding=1,
                                pool_size=2,
                                pool_type='max',
                                pool_stride=2)

            t2 = conv_relu_pool(input=t1,
                                conv_num_filters=164,
                                conv_filter_size=3,
                                conv_stride=1,
                                conv_padding=1,
                                pool_size=2,
                                pool_type='max',
                                pool_stride=2)

            t3 = conv_relu_pool(input=t2,
                                conv_num_filters=192,
                                conv_filter_size=3,
                                conv_stride=1,
                                conv_padding=1,
                                pool_size=2,
                                pool_type='max',
                                pool_stride=2)

            t4 = conv_relu_pool(input=t3,
                                conv_num_filters=192,
                                conv_filter_size=3,
                                conv_stride=1,
                                conv_padding=1,
                                pool_size=2,
                                pool_type='max',
                                pool_stride=2)

            t5 = conv_relu_pool(input=t4,
                                conv_num_filters=128,
                                conv_filter_size=3,
                                conv_stride=1,
                                conv_padding=1,
                                pool_size=3,
                                pool_type='max',
                                pool_stride=1,
                                pool_padding=1) # if no padding, it can not work

            fc1 = fluid.layers.fc(input=t5, size=128, act='relu')
            fc2 = fluid.layers.fc(input=fc1, size=config.class_dim, act='softmax') #logsoftmax
            #fc2 = fluid.layers.log(fc2) 
            return fc2

        h_f1, h_b1 = context_modeling(seq1, 'seq1')
        h_f2, h_b2 = context_modeling(seq2, 'seq2')
        simCube = pairwise_word_interaction_modeling(h_f1, h_b1, h_f2, h_b2)
        focusCube = similarity_focus_layer(simCube)
        # focusCube = fluid.layers.batch_norm(focusCube) # 0.8543 -> 0.8395
        prediction = similarity_classification_with_deep_convolutional_neural_networks(focusCube)

        #TODO: loss function
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=loss)
        acc = fluid.layers.accuracy(input=prediction, label=label)
        return avg_cost, acc, prediction
