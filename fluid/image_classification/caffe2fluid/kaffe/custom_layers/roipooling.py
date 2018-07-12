""" a custom layer for 'ROIPooling', maybe we should implement this in standard way.
    more info can be found here: http://caffe.berkeleyvision.org/tutorial/layers/ROIPooling.html
"""
from .register import register


def roipooling_shape(input_shapes, pooled_h, pooled_w, spatial_scale):
    """ calculate the output shape of this layer using input shape

    Args:
        @input_shape (list of num): a list of number which represents the input shape
        @out_max_val (bool): parameter from caffe's ROIPooling layer
        @top_k (int): parameter from caffe's ROIPooling layer
        @axis (int): parameter from caffe's ROIPooling layer

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """
    assert len(input_shapes) == 2, "not valid input shape for roipooling layer"
    base_fea_shape = input_shapes[0]
    rois_shape = input_shapes[1]
    output_shape = base_fea_shape
    output_shape[0] = rois_shape[0]
    output_shape[2] = pooled_h
    output_shape[3] = pooled_w
    return output_shape


def roipooling_layer(inputs, name, pooled_h, pooled_w, spatial_scale):
    """ build a layer of type 'ROIPooling' using fluid

    Args:
        @input (variable): input fluid variable for this layer
        @name (str): name for this layer
        @out_max_val (bool): parameter from caffe's ROIPooling layer
        @top_k (int): parameter from caffe's ROIPooling layer
        @axis (int): parameter from caffe's ROIPooling layer

    Returns:
        output (variable): output variable for this layer
    """

    import paddle.fluid as fluid
    assert len(inputs) == 2, "not valid input shape for roipooling layer"
    base_fea = inputs[0]
    rois = inputs[1][:, 1:5]
    rois_fea = fluid.layers.roi_pool(base_fea, rois, pooled_h, pooled_w,
                                     spatial_scale)

    return rois_fea


register(kind='ROIPooling', shape=roipooling_shape, layer=roipooling_layer)
