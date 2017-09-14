from paddle import v2 as paddle
from paddle.v2 import layer
from paddle.v2 import evaluator
from paddle.v2.activation import Relu, Linear
from paddle.v2.networks import img_conv_group, simple_gru


def conv_groups(input_image, num, with_bn):
    '''
    a deep CNN.
    @input_image: input image
    @num: number of CONV filters
    @with_bn: whether with batch normal
    '''
    assert num % 4 == 0

    tmp = img_conv_group(
        input=input_image,
        num_channels=1,
        conv_padding=1,
        conv_num_filter=[16] * (num / 4),
        conv_filter_size=3,
        conv_act=Relu(),
        conv_with_batchnorm=with_bn,
        pool_size=2,
        pool_stride=2, )

    tmp = img_conv_group(
        input=tmp,
        conv_padding=1,
        conv_num_filter=[32] * (num / 4),
        conv_filter_size=3,
        conv_act=Relu(),
        conv_with_batchnorm=with_bn,
        pool_size=2,
        pool_stride=2, )

    tmp = img_conv_group(
        input=tmp,
        conv_padding=1,
        conv_num_filter=[64] * (num / 4),
        conv_filter_size=3,
        conv_act=Relu(),
        conv_with_batchnorm=with_bn,
        pool_size=2,
        pool_stride=2, )

    tmp = img_conv_group(
        input=tmp,
        conv_padding=1,
        conv_num_filter=[128] * (num / 4),
        conv_filter_size=3,
        conv_act=Relu(),
        conv_with_batchnorm=with_bn,
        pool_size=2,
        pool_stride=2, )

    return tmp


class Model(object):
    def __init__(self, num_classes, shape, is_infer=False):
        '''
        @num_classes: int
            size of the character dict
        @shape: tuple of 2 int
            size of the input images
        '''
        self.num_classes = num_classes
        self.shape = shape
        self.is_infer = is_infer
        self.image_vector_size = shape[0] * shape[1]

        self.__declare_input_layers__()
        self.__build_nn__()

    def __declare_input_layers__(self):
        # image input as a float vector
        self.image = layer.data(
            name='image',
            type=paddle.data_type.dense_vector(self.image_vector_size),
            height=self.shape[0],
            width=self.shape[1])

        # label input as a ID list
        if self.is_infer == False:
            self.label = layer.data(
                name='label',
                type=paddle.data_type.integer_value_sequence(self.num_classes))

    def __build_nn__(self):
        # CNN output image features, 128 float matrixes
        conv_features = conv_groups(self.image, 8, True)

        # cutting CNN output into a sequence of feature vectors, which are
        # 1 pixel wide and 11 pixel high.
        sliced_feature = layer.block_expand(
            input=conv_features,
            num_channels=128,
            stride_x=1,
            stride_y=1,
            block_x=1,
            block_y=11)

        # RNNs to capture sequence information forwards and backwards.
        gru_forward = simple_gru(input=sliced_feature, size=128, act=Relu())
        gru_backward = simple_gru(
            input=sliced_feature, size=128, act=Relu(), reverse=True)

        # map each step of RNN to character distribution.
        self.output = layer.fc(
            input=[gru_forward, gru_backward],
            size=self.num_classes + 1,
            act=Linear())

        self.log_probs = paddle.layer.mixed(
            input=paddle.layer.identity_projection(input=self.output),
            act=paddle.activation.Softmax())

        # warp CTC to calculate cost for a CTC task.
        if self.is_infer == False:
            self.cost = layer.warp_ctc(
                input=self.output,
                label=self.label,
                size=self.num_classes + 1,
                norm_by_times=True,
                blank=self.num_classes)
