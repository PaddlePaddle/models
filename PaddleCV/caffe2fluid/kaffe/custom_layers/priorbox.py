""" A custom layer for 'priorbox' which is used in ssd to generate prior box info
    Since the order of prior box is different between caffe and paddle,
    we use 'slice' and 'concate' ops to align them.
"""

from .register import register


def priorbox_shape(input_shapes, min_size, max_size=None, aspect_ratio=None):
    """ calculate the output shape of this layer using input shapes

    Args:
        @input_shapes (list of tuples): a list of input shapes

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """
    assert len(input_shapes) == 2, "invalid inputs for Priorbox[%s]" % (name)
    fc_shape = input_shapes[0]
    N = 1
    if not max_size == None:
        N += 1
    if not aspect_ratio == None:
        N += 2 * len(aspect_ratio)

    N_bbx = fc_shape[2] * fc_shape[3] * N
    output_shape = [1, 2, 4 * N_bbx]
    return output_shape


def priorbox_layer(inputs,
                   name,
                   min_size,
                   max_size=None,
                   aspect_ratio=None,
                   variance=[0.1, 0.1, 0.2, 0.2],
                   flip=False,
                   clip=False,
                   step=0.0,
                   offset=0.5):
    """ build a layer of type 'Priorbox' using fluid

    Args:
        @inputs (list of variables): input fluid variables for this layer
        @name (str): name for this layer

    Returns:
        output (variable): output variable for this layer
    """
    import paddle.fluid as fluid

    assert len(inputs) == 2, "invalid inputs for Priorbox[%s]" % (name)
    input = inputs[0]
    image = inputs[1]
    steps = tuple(step) if type(step) is list or type(step) is tuple else (step,
                                                                           step)
    box, variance_ = fluid.layers.prior_box(
        input,
        image,
        min_size,
        max_size,
        aspect_ratio,
        variance,
        flip,
        clip,
        steps,
        offset,
        min_max_aspect_ratios_order=True)
    """
    #adjust layout when the output is not consistent with caffe's

    feat_shape = list(input.shape)
    H = feat_shape[2]
    W = feat_shape[3]
    box_tmp = fluid.layers.reshape(box, [H, W, -1, 4])
    nb_prior_bbx = int(box_tmp.shape[2])
    tensor_list = fluid.layers.split(box_tmp, nb_prior_bbx, 2)

    #TODO:
    #   current implementation for this layer is not efficient
    #   and we should fix this bug in future when Paddle support the same prior-box layout with Caffe
    index_list = [0]
    index_list = index_list * nb_prior_bbx
    index_offset = 0
    if max_size is not None:
        index_list[1] = -1
        index_offset = 1
    for ii in xrange(2 * len(aspect_ratio)):
        index_list[ii + 1 + index_offset] = ii + 1

    tensor_list_gathered = [tensor_list[ii] for ii in index_list]
    caffe_prior_bbx = fluid.layers.concat(tensor_list_gathered, axis=2)
    box = fluid.layers.reshape(caffe_prior_bbx, [1, 1, -1])
    """

    box = fluid.layers.reshape(box, [1, 1, -1])
    variance_ = fluid.layers.reshape(variance_, [1, 1, -1])
    output = fluid.layers.concat([box, variance_], axis=1)

    return output


register(kind='PriorBox', shape=priorbox_shape, layer=priorbox_layer)
