from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from src.models.libs.model_libs import scope, name_scope
from src.models.libs.model_libs import avg_pool, conv, bn, bn_zero, conv1d, FCNHead
from src.models.backbone.resnet import ResNet as resnet_backbone
from src.utils.config import cfg


def get_logit_interp(input, num_classes, out_shape, name="logit"):
    # 1x1_Conv
    param_attr = fluid.ParamAttr(
        name= name + 'weights',
        regularizer= fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0),
        initializer= fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))

    with scope(name):
        logit = conv(input, num_classes, filter_size=1, param_attr=param_attr, bias_attr=True, name=name+'_conv')
        logit_interp = fluid.layers.resize_bilinear( logit, out_shape=out_shape, name=name+'_interp')
    return logit_interp


def gcn_module(name_scope, x, num_node, num_state):
    '''
    input: any tensor of 3D, B,C,N
    '''
    print(x.shape)
    h = fluid.layers.transpose(x, perm=[0, 2, 1]) #B,C,N-->B,N,C
    h = conv1d(h, num_node, name_scope+'_conv1d1', bias_attr=True)
    h = fluid.layers.transpose(h, perm=[0, 2, 1]) #B,C,N
    h = fluid.layers.elementwise_add(h, x, act='relu')
    h = conv1d(h, num_state, name_scope+'_conv1d2', bias_attr= False)
    return h

def gru_module(x, num_state, num_node):
    '''
    Global Reasoning Unit: projection --> graph reasoning --> reverse projection
    params:
         x:  B x C x H x W
         num_state: the dimension of each vertex feature
         num_node: the number of vertet
    output: B x C x H x W
    feature trans:
            B, C, H, W --> B, N, H, W -->             B, N, H*W -->B, N, C1 -->B, C1, N-->B, C1, N-->B, C1, H*W-->B, C, H, W
                       --> B, C1,H, W -->B, C1,H*W -->B, H*W, C1
    '''
    # generate B
    num_batch, C, H, W = x.shape
    with scope('projection'):
        B = conv(x, num_node,
                filter_size=1,
                bias_attr=True,
                name='projection'+'_conv') #num_batch, node, H, W
        B = fluid.layers.reshape(B, shape=[num_batch, num_node, H*W]) # Projection Matrix: num_batch, node, L=H*W
    # reduce dimension
    with scope('reduce_channel'):
        x_reduce = conv(x, num_state,
                filter_size=1,
                bias_attr=True,
                name='reduce_channel'+'_conv') #num_batch, num_state, H, W
        x_reduce = fluid.layers.reshape(x_reduce, shape=[num_batch, num_state, H*W]) #num_batch, num_state, L
        x_reduce = fluid.layers.transpose(x_reduce, perm=[0, 2, 1]) #num_batch, L, num_state

    V = fluid.layers.transpose(fluid.layers.matmul(B, x_reduce), perm=[0,2,1]) #num_batch, num_state, num_node
    #L = fluid.layers.fill_constant(shape=[1], value=H*W, dtype='float32')
    #V = fluid.layers.elementwise_div(V, L)
    new_V = gcn_module('gru'+'_gcn', V, num_node, num_state)

    B = fluid.layers.reshape(B, shape= [num_batch, num_node, H*W])
    D = fluid.layers.transpose(B, perm=[0, 2, 1])
    Y = fluid.layers.matmul(D, fluid.layers.transpose(new_V, perm=[0, 2, 1]))
    Y = fluid.layers.transpose(Y, perm=[0, 2, 1])
    Y = fluid.layers.reshape(Y, shape=[num_batch, num_state, H, W])
    with scope('extend_dim'):
        Y = conv(Y, C, filter_size=1, bias_attr=False, name='extend_dim'+'_conv')
        #Y = bn_zero(Y)
        Y = bn(Y)
    out = fluid.layers.elementwise_add(Y, x)
    return out

def resnet(input):
    # end_points: end_layer of resnet backbone 
    # dilation_dict: dilation factor for stages_key
    scale = cfg.MODEL.GLORE.DEPTH_MULTIPLIER
    layers = cfg.MODEL.BACKBONE_LAYERS
    end_points = layers - 1
    dilation_dict = {2:2, 3:4}
    decode_points= [91, 100]
    model = resnet_backbone(layers, scale)
    res5, feat_dict = model.net(input,
                                end_points=end_points,
                                dilation_dict=dilation_dict,
                                decode_points= decode_points)

    return res5, feat_dict

def glore(input, num_classes):
    """
    Reference:
       Chen, Yunpeng, et al. "Graph-Based Global Reasoning Networks", In CVPR 2019
    """

    # Backbone: ResNet
    res5, feat_dict = resnet(input)
    res4= feat_dict[91]
    # 3x3 Conv. 2048 -> 512
    reduce_kernel=3
    if cfg.DATASET.DATASET_NAME=='cityscapes':
        reduce_kernel=1
    with scope('feature'):
        feature = conv(res5, 512, filter_size=reduce_kernel, bias_attr=False, name='feature_conv')
        feature = bn(feature, act='relu')
    # GRU Module
    gru_output = gru_module(feature,  num_state= 128,  num_node = 64)
    dropout = fluid.layers.dropout(gru_output, dropout_prob=0.1, name="dropout")

    logit = get_logit_interp(dropout, num_classes, input.shape[2:])
    if cfg.MODEL.GLORE.AuxHead:
        aux_logit = FCNHead(res4, 256, num_classes, input.shape[2:])
        return logit, aux_logit

    return logit

