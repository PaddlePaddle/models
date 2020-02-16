from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import paddle
import paddle.fluid as fluid
from src.utils.config import cfg
from src.models.libs.model_libs import scope, name_scope
from src.models.libs.model_libs import bn, bn_relu, relu, FCNHead
from src.models.libs.model_libs import conv
from src.models.libs.model_libs import separate_conv
from src.models.backbone.mobilenet_v2 import MobileNetV2 as mobilenet_backbone
from src.models.backbone.xception import Xception as xception_backbone
from src.models.backbone.resnet import ResNet as resnet_backbone
from src.models.backbone.hrnet import HRNet as hrnet_backbone



def ASPPHead(input, mid_channel, num_classes, output_shape):
    # Arch of Atorus Spatial Pyramid Pooling Module:                                                 
    #
    #          |----> ImagePool + Conv_1x1 + BN + ReLU + bilinear_interp-------->|————————|
    #          |                                                                 |        |
    #          |---->           Conv_1x1 + BN + ReLU                    -------->|        | 
    #          |                                                                 |        |
    #   x----->|---->        AtrousConv_3x3 + BN + ReLU                 -------->| concat |----> Conv_1x1 + BN + ReLU -->Dropout --> Conv_1x1 
    #          |                                                                 |        |
    #          |---->        AtrousConv_3x3 + BN + ReLU                 -------->|        |
    #          |                                                                 |        |
    #          |---->        AtorusConv_3x3 + BN + ReLU                 -------->|________|
    #                                                                                    
    #

    if cfg.MODEL.BACKBONE_OUTPUT_STRIDE == 16:
        aspp_ratios = [6, 12, 18]
    elif cfg.MODEL.BACKBONE_OUTPUT_STRIDE == 8:
        aspp_ratios = [12, 24, 36]
    else:
        raise Exception("deeplab only support stride 8 or 16")

    param_attr = fluid.ParamAttr(name=name_scope + 'weights', regularizer=None,
                                 initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.06))
    with scope('ASPPHead'):
        with scope("image_pool"):
            image_avg = fluid.layers.reduce_mean( input, [2, 3], keep_dim=True)
            image_avg = bn_relu( conv( image_avg, mid_channel, 1, 1, groups=1, padding=0, param_attr=param_attr))
            image_avg = fluid.layers.resize_bilinear(image_avg, input.shape[2:])

        with scope("aspp0"):
            aspp0 = bn_relu( conv( input, mid_channel, 1, 1, groups=1, padding=0, param_attr=param_attr))
        with scope("aspp1"):
            if cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV:
                aspp1 = separate_conv( input, mid_channel, 1, 3, dilation=aspp_ratios[0], act=relu)
            else:
                aspp1 = bn_relu( conv( input, mid_channel, stride=1, filter_size=3, dilation=aspp_ratios[0], 
                                       padding=aspp_ratios[0], param_attr=param_attr))
        with scope("aspp2"):
            if cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV:
                aspp2 = separate_conv( input, mid_channel, 1, 3, dilation=aspp_ratios[1], act=relu)
            else:
                aspp2 = bn_relu( conv( input, mid_channel, stride=1, filter_size=3, dilation=aspp_ratios[1], 
                                       padding=aspp_ratios[1], param_attr=param_attr))
        with scope("aspp3"):
            if cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV:
                aspp3 = separate_conv( input, mid_channel, 1, 3, dilation=aspp_ratios[2], act=relu)
            else:
                aspp3 = bn_relu( conv( input, mid_channel, stride=1, filter_size=3, dilation=aspp_ratios[2],
                                       padding=aspp_ratios[2], param_attr=param_attr))
        with scope("concat"):
            feat = fluid.layers.concat([image_avg, aspp0, aspp1, aspp2, aspp3], axis=1)
            feat = bn_relu( conv( feat, 2*mid_channel, 1, 1, groups=1, padding=0, param_attr=param_attr))
            feat = fluid.layers.dropout(feat, 0.1)

    # Conv_1x1 + bilinear_upsample
    seg_name = "logit"
    with scope(seg_name):
        param_attr = fluid.ParamAttr( name= seg_name+'_weights',
                                      regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0),
                                      initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))
        logit = conv(feat, num_classes, filter_size=1, param_attr=param_attr, bias_attr=True, name=seg_name+'_conv')
        logit_interp = fluid.layers.resize_bilinear(logit, out_shape=output_shape, name=seg_name+'_interp')

    return logit_interp



def mobilenetv2(input):
    # Backbone: mobilenetv2结构配置
    # DEPTH_MULTIPLIER: mobilenetv2的scale设置，默认1.0
    # OUTPUT_STRIDE：下采样倍数
    # end_points: mobilenetv2的block数
    # decode_point: 从mobilenetv2中引出分支所在block数, 作为decoder输入
    scale = cfg.MODEL.DEEPLABv3.DEPTH_MULTIPLIER
    output_stride = cfg.MODEL.DEEPLABv3.OUTPUT_STRIDE
    model = mobilenet_backbone(scale=scale, output_stride=output_stride)
    end_points = 18
    decode_point = 4
    data, decode_shortcuts = model.net(
        input, end_points=end_points, decode_points=decode_point)
    decode_shortcut = decode_shortcuts[decode_point]
    return data, decode_shortcut


def xception(input):
    # Backbone: Xception结构配置, xception_65, xception_41, xception_71三种可选
    # decode_point: 从Xception中引出分支所在block数，作为decoder输入
    # end_point：Xception的block数
    cfg.MODEL.DEFAULT_EPSILON = 1e-3
    model = xception_backbone(cfg.MODEL.BACKBONE)
    backbone = cfg.MODEL.BACKBONE
    output_stride = cfg.MODEL.DEEPLABv3.OUTPUT_STRIDE
    if '65' in backbone:
        decode_point = 2
        end_points = 21
    if '41' in backbone:
        decode_point = 2
        end_points = 13
    if '71' in backbone:
        decode_point = 3
        end_points = 23
    data, decode_shortcuts = model.net(
        input,
        output_stride=output_stride,
        end_points=end_points,
        decode_points=decode_point)
    decode_shortcut = decode_shortcuts[decode_point]
    return data, decode_shortcut


def resnet(input):
    # dilation_dict: 
    #     key: stage num
    #     value: dilation factor

    scale = cfg.MODEL.DEEPLABv3.DEPTH_MULTIPLIER
    layers = cfg.MODEL.BACKBONE_LAYERS
    end_points = layers - 1
    decode_points = [91,100 ]  # [10, 22, 91, 100], for obtaining feature maps of res2,res3, res4, and res5
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

def deeplabv3(input, num_classes):
    """
       Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation", in arXiv:1706:05587
    """
    if 'xception' in cfg.MODEL.BACKBONE:
        data, decode_shortcut = xception(input)
    elif 'mobilenet' in cfg.MODEL.BACKBONE:
        data, decode_shortcut = mobilenetv2(input)
    elif 'resnet' in cfg.MODEL.BACKBONE:
        res5, feat_dict = resnet(input)
        res4 = feat_dict[91]
    elif 'hrnet' in cfg.MODEL.BACKBONE:
        res5 = hrnet(input)
    else:
        raise Exception("deeplabv3 only support xception, mobilenet, resnet, and hrnet backbone")

    logit = ASPPHead(res5, mid_channel= 256, num_classes= num_classes, output_shape= input.shape[2:])
    if cfg.MODEL.DEEPLABv3.AuxHead:
        aux_logit = FCNHead(res4, 256, num_classes, input.shape[2:])
        return logit, aux_logit
    return logit

