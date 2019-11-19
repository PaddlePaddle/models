import paddle.fluid as fluid
from ActivationFunction import GaussActivation
from ActivationFunction import MaskUpdate


def ReverseMaskConv(inputMasks,
                    num_filters,
                    kSize=4,
                    stride=2,
                    padding=1,
                    convBias=False):
    maskFeatures = fluid.layers.conv2d(
        input=inputMasks,
        num_filters=num_filters,
        filter_size=kSize,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=convBias)

    maskActiv = GaussActivation(maskFeatures, 1.1, 2.0, 1.0, 1.0)

    maskUpdate = MaskUpdate(maskFeatures, 0.8)

    return maskActiv, maskUpdate




def ReverseAttention(ecFeaturesSkip, dcFeatures, maskFeaturesForAttention, num_filters, bn=True, activ='leaky', \
        kSize=4, stride=2, padding=1, outPadding=0,convBias=False):

    nextDcFeatures = fluid.layers.conv2d_transpose(
        input=dcFeatures,
        num_filters=num_filters,
        filter_size=kSize,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=convBias)
    concatFeatures = fluid.layers.concat(
        [ecFeaturesSkip, nextDcFeatures], axis=1)
    outputFeatures = concatFeatures * maskFeaturesForAttention

    if bn:
        outputFeatures = fluid.layers.batch_norm(input=outputFeatures)

    if activ == 'leaky':
        outputFeatures = fluid.layers.leaky_relu(outputFeatures, alpha=0.2)

    elif activ == 'relu':
        outputFeatures = fluid.layers.relu(outputFeatures)

    elif activ == 'sigmoid':
        outputFeatures = fluid.layers.sigmoid(outputFeatures)

    elif activ == 'tanh':
        outputFeatures = fluid.layers.tanh(outputFeatures)

    elif activ == 'prelu':
        outputFeatures = fluid.layers.prelu(outputFeatures, 'all')

    else:
        pass

    return outputFeatures
