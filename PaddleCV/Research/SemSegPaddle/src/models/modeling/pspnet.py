from __future__ import division
from __future__ import print_function
import sys
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from src.models.libs.model_libs import scope, name_scope
from src.models.libs.model_libs import avg_pool, conv, bn, FCNHead
from src.models.backbone.resnet import ResNet as resnet_backbone
from src.models.backbone.hrnet import HRNet as hrnet_backbone
from src.utils.config import cfg


def PSPHead(input, out_features, num_classes, output_shape):
    # Arch of Pyramid Scene Parsing Module:                                                 
    #
    #          |----> Pool_1x1 + Conv_1x1 + BN + ReLU + bilinear_interp-------->|————————|
    #          |                                                                |        |
    #          |----> Pool_2x2 + Conv_1x1 + BN + ReLU + bilinear_interp-------->|        | 
    # x ------>|                                                                | concat |----> Conv_3x3 + BN + ReLU -->Dropout --> Conv_1x1
    #     |    |----> Pool_3x3 + Conv_1x1 + BN + ReLU + bilinear_interp-------->|        | 
    #     |    |                                                                |        |
    #     |    |----> Pool_6x6 + Conv_1x1 + BN + ReLU + bilinear_interp-------->|________|
    #     |                                                                              ^
    #     |——————————————————————————————————————————————————————————————————————————————|
    #
    cat_layers = []
    sizes = (1,2,3,6)
    # 4 parallel pooling branches
    for size in sizes:
        psp_name = "psp" + str(size)
        with scope(psp_name):
            pool_feat = fluid.layers.adaptive_pool2d(input, pool_size=[size, size], pool_type='avg', 
                                                name=psp_name+'_adapool')
            conv_feat = conv(pool_feat, out_features, filter_size=1, bias_attr=True, 
                        name= psp_name + '_conv')
            bn_feat = bn(conv_feat, act='relu')
            interp = fluid.layers.resize_bilinear(bn_feat, out_shape=input.shape[2:], name=psp_name+'_interp') 
        cat_layers.append(interp)
    cat_layers = [input] + cat_layers[::-1]
    cat = fluid.layers.concat(cat_layers, axis=1, name='psp_cat')
    
    # Conv_3x3 + BN + ReLU
    psp_end_name = "psp_end"
    with scope(psp_end_name):
        data = conv(cat, out_features, filter_size=3, padding=1, bias_attr=True, name=psp_end_name)
        out = bn(data, act='relu')
    # Dropout
    dropout_out = fluid.layers.dropout(out, dropout_prob=0.1, name="dropout")
   
    # Conv_1x1 + bilinear_upsample
    seg_name = "logit"
    with scope(seg_name):
        param_attr = fluid.ParamAttr( name= seg_name+'_weights',
                                      regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0),
                                      initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))
        logit = conv(dropout_out, num_classes, filter_size=1, param_attr=param_attr, bias_attr=True, name=seg_name+'_conv')
        logit_interp = fluid.layers.resize_bilinear(logit, out_shape=output_shape, name=seg_name+'_interp') 

    return logit_interp

def resnet(input):
    # dilation_dict: 
    #     key: stage num
    #     value: dilation factor

    scale = cfg.MODEL.PSPNET.DEPTH_MULTIPLIER
    layers = cfg.MODEL.BACKBONE_LAYERS
    end_points = layers - 1
    decode_points = [91, 100]  # [10, 22, 91, 100], for obtaining feature maps of res2,res3, res4, and res5
    dilation_dict = {2:2, 3:4}
    model = resnet_backbone(layers, scale)
    res5, feat_dict = model.net(input, 
                                end_points=end_points, 
                                dilation_dict=dilation_dict,
                                decode_points=decode_points)
    return res5, feat_dict

def hrnet(input):
    model = hrnet_backbone(stride=4, seg_flag=True)
    feats = model.net(input)
    return feats

def pspnet(input, num_classes):
    """
    Reference: 
        Zhao, Hengshuang, et al. "Pyramid scene parsing network.", In CVPR 2017
    """
    if 'resnet' in cfg.MODEL.BACKBONE:
        res5, feat_dict = resnet(input)
        res4 = feat_dict[91]
    elif 'hrnet' in cfg.MODEL.BACKBONE:
        res5 = hrnet(input)
    else:
        raise Exception("pspnet only support resnet and hrnet backbone")
    logit = PSPHead(res5, 512, num_classes, input.shape[2:])
    if cfg.MODEL.PSPNET.AuxHead:
        aux_logit = FCNHead(res4, 256, num_classes, input.shape[2:])
        return logit, aux_logit
    return logit

