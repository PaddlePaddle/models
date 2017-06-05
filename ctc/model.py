from paddle import v2 as paddle
from paddle.v2 import layer
from paddle.v2.activation import Relu, Linear
from paddle.v2.networks import img_conv_group, simple_gru
from paddle.v2 import networks

from data_provider import AsciiDic, ImageDataset

num_classes = AsciiDic().size()
image_shape = (173 / 2, 46 / 2)
image_vector_size = image_shape[0] * image_shape[1]
image_file_list = '/home/disk1/yanchunwei/90kDICT32px/train_all.txt'
trainer_count = 4
batch_size = 30 * trainer_count

paddle.init(use_gpu=True, trainer_count=trainer_count)


def ocr_convs(input_image, num, with_bn):
    '''
    @input_image: input image
    '''
    assert num % 4 == 0

    tmp = input_image
    # for num_filter in [16, 32, 64, 128]:
    #     tmp = img_conv_group(
    #         input=tmp,
    #         num_channels=1,
    #         conv_padding=1,
    #         conv_num_filter=[16] * (num / 4),
    #         conv_filter_size=3,
    #         conv_act=Relu(),
    #         conv_with_batchnorm=with_bn,
    #         pool_size=2,
    #         pool_stride=2)
    #     break

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

    # tmp = img_conv_group(
    #       input = tmp,
    #       conv_padding = 1,
    #       conv_num_filter = [128] * (num / 4),
    #       conv_filter_size = 3,
    #       conv_act = Relu(),
    #       conv_with_batchnorm = with_bn,
    #       pool_size = 2,
    #       pool_stride = 2,
    #       )

    # tmp = img_conv_group(
    #     input=tmp,
    #     conv_padding=1,
    #     conv_num_filter=[32] * (num / 4),
    #     conv_filter_size=3,
    #     conv_act=Relu(),
    #     conv_with_batchnorm=with_bn,
    #     pool_size=2,
    #     pool_stride=2, )

    # tmp = img_conv_group(
    #     input=tmp,
    #     conv_padding=1,
    #     conv_num_filter=[64] * (num / 4),
    #     conv_filter_size=3,
    #     conv_act=Relu(),
    #     conv_with_batchnorm=with_bn,
    #     pool_size=2,
    #     pool_stride=2, )

    # tmp = img_conv_group(
    #     input=tmp,
    #     conv_padding=1,
    #     conv_num_filter=[128] * (num / 4),
    #     conv_filter_size=3,
    #     conv_act=Relu(),
    #     conv_with_batchnorm=with_bn,
    #     pool_size=2,
    #     pool_stride=2, )
    return tmp


# ==============================================================================
#                    input layers
# ==============================================================================

image = layer.data(
    name='image',
    type=paddle.data_type.dense_vector(image_vector_size),
    height=image_shape[0],
    width=image_shape[1])

label = layer.data(
    name='label',
    type=paddle.data_type.integer_value_sequence(AsciiDic().size()))

# ==============================================================================
#                    model structure
# ==============================================================================
conv_features = ocr_convs(image, 8, True)

sliced_feature = layer.block_expand(
    input=conv_features,
    num_channels=64,
    stride_x=1,
    stride_y=1,
    block_x=1,
    block_y=11)

gru_forward = simple_gru(input=sliced_feature, size=128, act=Relu())
gru_backward = simple_gru(
    input=sliced_feature, size=128, act=Relu(), reverse=True)

output = layer.fc(
    input=[gru_forward, gru_backward], size=num_classes + 1, act=Linear())

cost = layer.warp_ctc(
    input=output,
    label=label,
    size=num_classes + 1,
    norm_by_times=True,
    blank=num_classes)

params = paddle.parameters.create(cost)

optimizer = paddle.optimizer.Momentum(momentum=0)

trainer = paddle.trainer.SGD(
    cost=cost, parameters=params, update_equation=optimizer)

from data_provider import get_file_list

dataset = ImageDataset(
    get_file_list(image_file_list),
    fixed_shape=image_shape, )


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 20 == 0:
            print "Pass %d, Samples %d, Cost %f" % (
                event.pass_id, event.batch_id * batch_size, event.cost)

        if event.batch_id > 0 and event.batch_id % 10 == 0:
            result = trainer.test(
                reader=paddle.batch(dataset.test, batch_size=50),
                feeding={'image': 0,
                         'label': 1})
            print "Test %d-%d, Cost %f" % (event.pass_id, event.batch_id,
                                           result.cost)


trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(dataset.train, buf_size=100),
        batch_size=batch_size),
    feeding={'image': 0,
             'label': 1},
    event_handler=event_handler,
    num_passes=1000)
