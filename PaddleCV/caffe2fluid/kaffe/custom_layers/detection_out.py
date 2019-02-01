""" A custom layer for 'detectionout' used in 'SSD' model to produce outputs
    Note: Since Paddle's implementation of 'detectionout' applied 'flatten' and 'softmax' ops on the input of 'conf', 
    while Caffe's implementation do not.
"""

from .register import register


def detectionoutput_shape(input_shape):
    """ the output shape of this layer is dynamic and not determined by 'input_shape'

    Args:
        @input_shape (list of int): input shape

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """
    output_shape = [-1, 6]
    return output_shape


def detectionoutput_layer(inputs,
                          name,
                          background_label=0,
                          share_location=True,
                          nms_param=None,
                          keep_top_k=100,
                          confidence_threshold=0.1):
    """ build a layer of type 'detectionout' using fluid

    Args:
        @inputs (list of variables): input fluid variables for this layer
        @name (str): name for this layer

    Returns:
        output (variable): output variable for this layer
    """
    import paddle.fluid as fluid

    if nms_param is None:
        nms_param = {"nms_threshold": 0.3, "top_k": 10, "eta": 1.0}

    mbox_conf_flatten = inputs[1]
    mbox_priorbox = inputs[2]
    mbox_priorbox_list = fluid.layers.split(mbox_priorbox, 2, dim=1)
    pb = mbox_priorbox_list[0]
    pbv = mbox_priorbox_list[1]
    pb = fluid.layers.reshape(x=pb, shape=[-1, 4])
    pbv = fluid.layers.reshape(x=pbv, shape=[-1, 4])
    mbox_loc = inputs[0]
    mbox_loc = fluid.layers.reshape(
        x=mbox_loc, shape=[-1, mbox_conf_flatten.shape[1], 4])

    default = {"nms_threshold": 0.3, "top_k": 10, "eta": 1.0}
    fields = ['eta', 'top_k', 'nms_threshold']

    for f in default.keys():
        if not nms_param.has_key(f):
            nms_param[f] = default[f]

    nmsed_outs = fluid.layers.detection_output(
        scores=mbox_conf_flatten,
        loc=mbox_loc,
        prior_box=pb,
        prior_box_var=pbv,
        background_label=background_label,
        nms_threshold=nms_param["nms_threshold"],
        nms_top_k=nms_param["top_k"],
        keep_top_k=keep_top_k,
        score_threshold=confidence_threshold,
        nms_eta=nms_param["eta"])

    return nmsed_outs


register(
    kind='DetectionOutput',
    shape=detectionoutput_shape,
    layer=detectionoutput_layer)
