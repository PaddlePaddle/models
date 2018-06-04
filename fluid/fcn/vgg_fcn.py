import paddle.v2 as paddle
import paddle.fluid as fluid
import pdb


def vgg16_fcn(input, args):
    '''The definition of FCN architecture.
    
    The definition of FCN architecture based on VGG16 network, we implemented FCN-32s, FCN-16s and
    FCN-8s currently.
    
    Args:
        input: The input data of network.
        args: The choice of network architecture.
    
    Returns:
        upscore: The output of network.
    '''

    def conv(input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=None,
             group=1,
             biased=True):
        if padding is None:
            padding = [0, 0]

        c_i, h_i, w_i = input.shape[1:]
        assert c_i % group == 0
        assert c_o % group == 0

        prefix = name + '_'
        output = fluid.layers.conv2d(
            input=input,
            filter_size=[k_h, k_w],
            num_filters=c_o,
            stride=[s_h, s_w],
            padding=padding,
            groups=group,
            param_attr=fluid.ParamAttr(
                name=prefix + 'weights',
                initializer=fluid.initializer.Xavier()),
            bias_attr=fluid.ParamAttr(
                name=prefix + 'biases',
                initializer=fluid.initializer.Constant(0)),
            act='relu' if relu is True else None)
        return output

    def pool(pool_type, input, k_h, k_w, s_h, s_w, name, padding):
        in_hw = input.shape[2:]
        k_hw = [k_h, k_w]
        s_hw = [s_h, s_w]

        output = fluid.layers.pool2d(
            input=input,
            pool_size=k_hw,
            pool_stride=s_hw,
            pool_padding=padding,
            ceil_mode=True,
            pool_type=pool_type)
        return output

    def max_pool(input, k_h, k_w, s_h, s_w, name, padding=[0, 0]):
        return pool('max', input, k_h, k_w, s_h, s_w, name, padding)

    conv1_1 = conv(input, 3, 3, 64, 1, 1, padding=[100, 100], name='conv1_1')
    conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, padding=[1, 1], name='conv1_2')
    pool1 = max_pool(conv1_2, 2, 2, 2, 2, name='pool1')
    conv2_1 = conv(pool1, 3, 3, 128, 1, 1, padding=[1, 1], name='conv2_1')
    conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, padding=[1, 1], name='conv2_2')
    pool2 = max_pool(conv2_2, 2, 2, 2, 2, name='pool2')
    conv3_1 = conv(pool2, 3, 3, 256, 1, 1, padding=[1, 1], name='conv3_1')
    conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, padding=[1, 1], name='conv3_2')
    conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, padding=[1, 1], name='conv3_3')
    pool3 = max_pool(conv3_3, 2, 2, 2, 2, name='pool3')
    conv4_1 = conv(pool3, 3, 3, 512, 1, 1, padding=[1, 1], name='conv4_1')
    conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, padding=[1, 1], name='conv4_2')
    conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, padding=[1, 1], name='conv4_3')
    pool4 = max_pool(conv4_3, 2, 2, 2, 2, name='pool4')
    conv5_1 = conv(pool4, 3, 3, 512, 1, 1, padding=[1, 1], name='conv5_1')
    conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, padding=[1, 1], name='conv5_2')
    conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, padding=[1, 1], name='conv5_3')
    pool5 = max_pool(conv5_3, 2, 2, 2, 2, name='pool5')

    fc6 = conv(pool5, 7, 7, 4096, 1, 1, name='fc6')
    fc7 = conv(fc6, 1, 1, 4096, 1, 1, name='fc7')
    score_fr = conv(fc7, 1, 1, 21, 1, 1, relu=False, name='score_fr')

    if args.fcn_arch == 'fcn-32s':
        upscore = fluid.layers.conv2d_transpose(
            input=score_fr,
            num_filters=21,
            filter_size=156,
            stride=16,
            param_attr=fluid.ParamAttr(
                name='upscore_weights', initializer=fluid.initializer.Xavier()),
            bias_attr=fluid.ParamAttr(
                name='upscore_biases',
                initializer=fluid.initializer.Constant(0)))
    elif args.fcn_arch == 'fcn-16s':
        upscore2 = fluid.layers.conv2d_transpose(
            input=score_fr,
            num_filters=21,
            filter_size=5,
            stride=3,
            param_attr=fluid.ParamAttr(
                name='upscore2_weights',
                initializer=fluid.initializer.Xavier()),
            bias_attr=fluid.ParamAttr(
                name='upscore2_biases',
                initializer=fluid.initializer.Constant(0)))
        score_pool4 = conv(pool4, 1, 1, 21, 1, 1, name='score_pool4')
        fuse_pool4 = fluid.layers.sums(input=[upscore2, score_pool4])

        upscore = fluid.layers.conv2d_transpose(
            input=fuse_pool4,
            num_filters=21,
            filter_size=52,
            stride=8,
            param_attr=fluid.ParamAttr(
                name='upscore_weights', initializer=fluid.initializer.Xavier()),
            bias_attr=fluid.ParamAttr(
                name='upscore_biases',
                initializer=fluid.initializer.Constant(0)))
    elif args.fcn_arch == 'fcn-8s':
        upscore2 = fluid.layers.conv2d_transpose(
            input=score_fr,
            num_filters=21,
            filter_size=5,
            stride=3,
            param_attr=fluid.ParamAttr(
                name='upscore2_weights',
                initializer=fluid.initializer.Xavier()),
            bias_attr=fluid.ParamAttr(
                name='upscore2_biases',
                initializer=fluid.initializer.Constant(0)))
        score_pool4 = conv(pool4, 1, 1, 21, 1, 1, name='score_pool4')
        fuse_pool4 = fluid.layers.sums(input=[upscore2, score_pool4])

        upscore1 = fluid.layers.conv2d_transpose(
            input=fuse_pool4,
            num_filters=21,
            filter_size=1,
            stride=2,
            param_attr=fluid.ParamAttr(
                name='upscore1_weights',
                initializer=fluid.initializer.Xavier()),
            bias_attr=fluid.ParamAttr(
                name='upscore1_biases',
                initializer=fluid.initializer.Constant(0)))
        score_pool3 = conv(pool3, 1, 1, 21, 1, 1, name='score_pool3')
        fuse_pool3 = fluid.layers.sums(input=[upscore1, score_pool3])

        upscore = fluid.layers.conv2d_transpose(
            input=fuse_pool3,
            num_filters=21,
            filter_size=52,
            stride=4,
            param_attr=fluid.ParamAttr(
                name='upscore_weights', initializer=fluid.initializer.Xavier()),
            bias_attr=fluid.ParamAttr(
                name='upscore_biases',
                initializer=fluid.initializer.Constant(0)))
    return upscore
