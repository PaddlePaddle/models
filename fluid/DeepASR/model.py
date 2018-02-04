from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.v2 as paddle
import paddle.v2.fluid as fluid


def stacked_lstmp_model(hidden_dim,
                        proj_dim,
                        stacked_num,
                        class_num=1749,
                        is_train=True):
    feature = fluid.layers.data(
        name="feature", shape=[-1, 120 * 11], dtype="float32", lod_level=1)

    seq_conv1 = fluid.layers.sequence_conv(
        input=feature,
        num_filters=1024,
        filter_size=3,
        filter_stride=1,
        bias_attr=True)
    bn1 = fluid.layers.batch_norm(
        input=seq_conv1,
        act="sigmoid",
        is_test=not is_train,
        momentum=0.9,
        epsilon=1e-05,
        data_layout='NCHW')

    stack_input = bn1
    for i in range(stacked_num):
        fc = fluid.layers.fc(input=stack_input,
                             size=hidden_dim * 4,
                             bias_attr=True)
        proj, cell = fluid.layers.dynamic_lstmp(
            input=fc,
            size=hidden_dim * 4,
            proj_size=proj_dim,
            bias_attr=True,
            use_peepholes=True,
            is_reverse=False,
            cell_activation="tanh",
            proj_activation="tanh")
        bn = fluid.layers.batch_norm(
            input=proj,
            act="sigmoid",
            is_test=not is_train,
            momentum=0.9,
            epsilon=1e-05,
            data_layout='NCHW')
        stack_input = bn

    prediction = fluid.layers.fc(input=stack_input,
                                 size=class_num,
                                 act='softmax')

    if not is_train: return feature, prediction

    label = fluid.layers.data(
        name="label", shape=[-1, 1], dtype="int64", lod_level=1)
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    return prediction, label, avg_cost
