import paddle.fluid as fluid
from ActivationFunction import GaussActivation
from ActivationFunction import MaskUpdate

# learnable forward attention conv layer


def ForwardAttentionLayer(inputFeatures,
                          inputMasks,
                          num_filters,
                          kSize,
                          stride,
                          padding,
                          bias=False):
    convFeatures = fluid.layers.conv2d(
        input=inputFeatures,
        num_filters=num_filters,
        filter_size=kSize,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias)
    maskFeatures = fluid.layers.conv2d(
        input=inputMasks,
        num_filters=num_filters,
        filter_size=kSize,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias)

    maskActiv = GaussActivation(maskFeatures, 1.1, 2.0, 1.0, 1.0)
    convOut = convFeatures * maskActiv

    maskUpdate = MaskUpdate(maskFeatures, 0.8)

    return convOut, maskUpdate, convFeatures, maskActiv


def ForwardAttention(inputFeatures,
                     inputMasks,
                     num_filters,
                     bn=True,
                     sample='down-4',
                     activ='leaky',
                     convBias=False):
    if sample == 'down-4':
        kSize = 4
        stride = 2
        padding = 1
    elif sample == 'down-5':
        kSize = 5
        stride = 2
        padding = 2

    elif sample == 'down-7':
        kSize = 7
        stride = 2
        padding = 3
    elif sample == 'down-3':
        kSize = 3
        stride = 2
        padding = 1
    else:
        kSize = 3
        stride = 1
        padding = 1
    features, maskUpdated, convPreF, maskActiv = ForwardAttentionLayer(
        inputFeatures,
        inputMasks,
        num_filters,
        kSize,
        stride,
        padding,
        bias=convBias)

    if bn:
        features = fluid.layers.batch_norm(input=features)

    if activ == 'leaky':
        features = fluid.layers.leaky_relu(features, alpha=0.2)

    elif activ == 'relu':
        features = fluid.layers.relu(features)

    elif activ == 'sigmoid':
        features = fluid.layers.sigmoid(features)

    elif activ == 'tanh':
        features = fluid.layers.tanh(features)

    elif activ == 'prelu':
        features = fluid.layers.prelu(features, 'all')

    else:
        pass

    return features, maskUpdated, convPreF, maskActiv
