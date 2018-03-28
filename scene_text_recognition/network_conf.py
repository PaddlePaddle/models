from paddle import v2 as paddle
from paddle.v2 import layer
from paddle.v2 import evaluator
from paddle.v2.activation import Relu, Linear
from paddle.v2.networks import img_conv_group, simple_gru
from config import ModelConfig as conf


class Model(object):
    def __init__(self, num_classes, shape, is_infer=False):
        '''
        :param num_classes: The size of the character dict.
        :type num_classes: int
        :param shape: The size of the input images.
        :type shape: tuple of 2 int
        :param is_infer: The boolean parameter indicating
                         inferring or training.
        :type shape: bool
        '''
        self.num_classes = num_classes
        self.shape = shape
        self.is_infer = is_infer
        self.image_vector_size = shape[0] * shape[1]

        self.__declare_input_layers__()
        self.__build_nn__()

    def __declare_input_layers__(self):
        '''
        Define the input layer.
        '''
        # Image input as a float vector.
        self.image = layer.data(
            name='image',
            type=paddle.data_type.dense_vector(self.image_vector_size),
            height=self.shape[0],
            width=self.shape[1])

        # Label input as an ID list
        if not self.is_infer:
            self.label = layer.data(
                name='label',
                type=paddle.data_type.integer_value_sequence(self.num_classes))

    def __build_nn__(self):
        '''
        Build the network topology.
        '''
        # Get the image features with CNN.
        conv_features = self.conv_groups(self.image, conf.filter_num,
                                         conf.with_bn)

        # Expand the output of CNN into a sequence of feature vectors.
        sliced_feature = layer.block_expand(
            input=conv_features,
            num_channels=conf.num_channels,
            stride_x=conf.stride_x,
            stride_y=conf.stride_y,
            block_x=conf.block_x,
            block_y=conf.block_y)

        # Use RNN to capture sequence information forwards and backwards.
        gru_forward = simple_gru(
            input=sliced_feature, size=conf.hidden_size, act=Relu())
        gru_backward = simple_gru(
            input=sliced_feature,
            size=conf.hidden_size,
            act=Relu(),
            reverse=True)

        # Map the output of RNN to character distribution.
        self.output = layer.fc(input=[gru_forward, gru_backward],
                               size=self.num_classes + 1,
                               act=Linear())

        self.log_probs = paddle.layer.mixed(
            input=paddle.layer.identity_projection(input=self.output),
            act=paddle.activation.Softmax())

        # Use warp CTC to calculate cost for a CTC task.
        if not self.is_infer:
            self.cost = layer.warp_ctc(
                input=self.output,
                label=self.label,
                size=self.num_classes + 1,
                norm_by_times=conf.norm_by_times,
                blank=self.num_classes)

            self.eval = evaluator.ctc_error(input=self.output, label=self.label)

    def conv_groups(self, input, num, with_bn):
        '''
        Get the image features with image convolution group.

        :param input: Input layer.
        :type input: LayerOutput
        :param num: Number of the filters.
        :type num: int
        :param with_bn: Use batch normalization or not.
        :type with_bn: bool
        '''
        assert num % 4 == 0

        filter_num_list = conf.filter_num_list
        is_input_image = True
        tmp = input

        for num_filter in filter_num_list:

            if is_input_image:
                num_channels = 1
                is_input_image = False
            else:
                num_channels = None

            tmp = img_conv_group(
                input=tmp,
                num_channels=num_channels,
                conv_padding=conf.conv_padding,
                conv_num_filter=[num_filter] * (num / 4),
                conv_filter_size=conf.conv_filter_size,
                conv_act=Relu(),
                conv_with_batchnorm=with_bn,
                pool_size=conf.pool_size,
                pool_stride=conf.pool_stride, )

        return tmp
