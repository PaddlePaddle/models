from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid


def stacked_lstmp_model(frame_dim,
                        hidden_dim,
                        proj_dim,
                        stacked_num,
                        class_num,
                        parallel=False,
                        is_train=True):
    """ The model for DeepASR. The main structure is composed of stacked 
        identical LSTMP (LSTM with recurrent projection) layers.

        When running in training and validation phase, the feeding dictionary
        is {'feature', 'label'}, fed by the LodTensor for feature data and 
        label data respectively. And in inference, only `feature` is needed.

    Args:
        frame_dim(int): The frame dimension of feature data.
        hidden_dim(int): The hidden state's dimension of the LSTMP layer.
        proj_dim(int): The projection size of the LSTMP layer.
        stacked_num(int): The number of stacked LSTMP layers.
        parallel(bool): Run in parallel or not, default `False`.
        is_train(bool): Run in training phase or not, default `True`.
        class_dim(int): The number of output classes.
    """

    # network configuration
    def _net_conf(feature, label):
        conv1 = fluid.layers.conv2d(
            input=feature,
            num_filters=32,
            filter_size=3,
            stride=1,
            padding=1,
            bias_attr=True,
            act="relu")

        pool1 = fluid.layers.pool2d(
            conv1, pool_size=3, pool_type="max", pool_stride=2, pool_padding=0)

        stack_input = pool1
        for i in range(stacked_num):
            fc = fluid.layers.fc(input=stack_input,
                                 size=hidden_dim * 4,
                                 bias_attr=None)
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
                is_test=not is_train,
                momentum=0.9,
                epsilon=1e-05,
                data_layout='NCHW')
            stack_input = bn

        prediction = fluid.layers.fc(input=stack_input,
                                     size=class_num,
                                     act='softmax')

        cost = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc = fluid.layers.accuracy(input=prediction, label=label)
        return prediction, avg_cost, acc

    # data feeder
    feature = fluid.layers.data(
        name="feature",
        shape=[-1, 3, 11, frame_dim],
        dtype="float32",
        lod_level=1)
    label = fluid.layers.data(
        name="label", shape=[-1, 1], dtype="int64", lod_level=1)

    if parallel:
        # When the execution place is specified to CUDAPlace, the program will
        # run on all $CUDA_VISIBLE_DEVICES GPUs. Otherwise the program will 
        # run on all CPU devices.
        places = fluid.layers.device.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            feat_ = pd.read_input(feature)
            label_ = pd.read_input(label)
            prediction, avg_cost, acc = _net_conf(feat_, label_)
            for out in [prediction, avg_cost, acc]:
                pd.write_output(out)

        # get mean loss and acc through every devices.
        prediction, avg_cost, acc = pd()
        prediction.stop_gradient = True
        avg_cost = fluid.layers.mean(x=avg_cost)
        acc = fluid.layers.mean(x=acc)
    else:
        prediction, avg_cost, acc = _net_conf(feature, label)

    return prediction, avg_cost, acc
