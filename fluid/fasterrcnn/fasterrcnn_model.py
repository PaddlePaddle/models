from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.regularizer import L2Decay
import paddle.fluid as fluid
import sys


def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  name=None):
    conv1 = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=ParamAttr(name=name + "_biases"),
        name=name + '.conv2d.output.1')
    if name == "conv1":
        bn_name = "bn_" + name
    else:
        bn_name = "bn" + name[3:]

    return fluid.layers.batch_norm(
        input=conv1,
        act=act,
        name=bn_name + '.output.1',
        param_attr=ParamAttr(name=bn_name + '_scale'),
        bias_attr=ParamAttr(bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance',
        is_test=True)


def conv_affine_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      name=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=ParamAttr(
            name=name + "_biases", learning_rate=2., regularizer=L2Decay(0.)),
        name=name + '.conv2d.output.1')
    if name == "conv1":
        bn_name = "bn_" + name
    else:
        bn_name = "bn" + name[3:]

    scale = fluid.layers.create_parameter(
        shape=[conv.shape[1]],
        dtype=conv.dtype,
        attr=ParamAttr(
            name=bn_name + '_scale', learning_rate=0.),
        default_initializer=Constant(1.))
    scale.stop_gradient = True
    bias = fluid.layers.create_parameter(
        shape=[conv.shape[1]],
        dtype=conv.dtype,
        attr=ParamAttr(
            bn_name + '_offset', learning_rate=0.),
        default_initializer=Constant(0.))
    bias.stop_gradient = True

    elt_mul = fluid.layers.elementwise_mul(x=conv, y=scale, axis=1)
    out = fluid.layers.elementwise_add(x=elt_mul, y=bias, axis=1)
    if act == 'relu':
        out = fluid.layers.relu(x=out)
    return out


def shortcut(input, ch_out, stride, name):
    ch_in = input.shape[1]  # if args.data_format == 'NCHW' else input.shape[-1]
    if ch_in != ch_out:
        return conv_affine_layer(input, ch_out, 1, stride, 0, None, name=name)
    else:
        return input


def basicblock(input, ch_out, stride, name):
    short = shortcut(input, ch_out, stride, name=name)
    conv1 = conv_affine_layer(input, ch_out, 3, stride, 1, name=name)
    conv2 = conv_affine_layer(conv1, ch_out, 3, 1, 1, act=None, name=name)
    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name)


def bottleneck(input, ch_out, stride, name):
    short = shortcut(input, ch_out * 4, stride, name=name + "_branch1")
    conv1 = conv_affine_layer(
        input, ch_out, 1, stride, 0, name=name + "_branch2a")
    conv2 = conv_affine_layer(conv1, ch_out, 3, 1, 1, name=name + "_branch2b")
    conv3 = conv_affine_layer(
        conv2, ch_out * 4, 1, 1, 0, act=None, name=name + "_branch2c")
    return fluid.layers.elementwise_add(
        x=short, y=conv3, act='relu', name=name + ".add.output.5")


def layer_warp(block_func, input, ch_out, count, stride, name):
    res_out = block_func(input, ch_out, stride, name=name + "a")
    for i in range(1, count):
        res_out = block_func(res_out, ch_out, 1, name=name + chr(ord("a") + i))
    return res_out


def FasterRcnn(input, depth, anchor_sizes, variance, aspect_ratios, gt_box,
               is_crowd, gt_label, im_info, class_nums, use_random):

    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]
    conv1 = conv_affine_layer(
        input, ch_out=64, filter_size=7, stride=2, padding=3, name="conv1")
    pool1 = fluid.layers.pool2d(
        input=conv1,
        pool_type='max',
        pool_size=3,
        pool_stride=2,
        pool_padding=1,
        name="pool1.max_pool.output.1")
    res2 = layer_warp(block_func, pool1, 64, stages[0], 1, name="res2")
    res3 = layer_warp(block_func, res2, 128, stages[1], 2, name="res3")
    res4 = layer_warp(block_func, res3, 256, stages[2], 2, name="res4")
    #========= RPN ============

    # rpn_conv/3*3
    rpn_conv = fluid.layers.conv2d(
        input=res4,
        num_filters=1024,
        filter_size=3,
        stride=1,
        padding=1,
        act='relu',
        name='conv_rpn',
        param_attr=ParamAttr(name="conv_rpn_w"),
        bias_attr=ParamAttr(
            name="conv_rpn_b", learning_rate=2., regularizer=L2Decay(0.)))
    anchor, var = fluid.layers.anchor_generator(
        input=rpn_conv,
        anchor_sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
        variance=variance,
        stride=[16.0, 16.0])

    num_anchor = anchor.shape[2]
    rpn_cls_score = fluid.layers.conv2d(
        rpn_conv,
        num_filters=num_anchor,
        filter_size=1,
        stride=1,
        padding=0,
        act=None,
        name='rpn_cls_score',
        param_attr=ParamAttr(name="rpn_cls_logits_w"),
        bias_attr=ParamAttr(
            name="rpn_cls_logits_b", learning_rate=2., regularizer=L2Decay(0.)))
    rpn_bbox_pred = fluid.layers.conv2d(
        rpn_conv,
        num_filters=4 * num_anchor,
        filter_size=1,
        stride=1,
        padding=0,
        act=None,
        name='rpn_bbox_pred',
        param_attr=ParamAttr(name="rpn_bbox_pred_w"),
        bias_attr=ParamAttr(
            name="rpn_bbox_pred_b", learning_rate=2., regularizer=L2Decay(0.)))

    rpn_cls_score_prob = fluid.layers.sigmoid(
        rpn_cls_score, name='rpn_cls_score_prob')

    rpn_rois, rpn_roi_probs = fluid.layers.generate_proposals(
        scores=rpn_cls_score_prob,
        bbox_deltas=rpn_bbox_pred,
        im_info=im_info,
        anchors=anchor,
        variances=var,
        pre_nms_top_n=12000,
        post_nms_top_n=2000,
        nms_thresh=0.7,
        min_size=0.0,
        eta=1.0)
    rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights = fluid.layers.generate_proposal_labels(
        rpn_rois=rpn_rois,
        gt_classes=gt_label,
        is_crowd=is_crowd,
        gt_boxes=gt_box,
        im_info=im_info,
        batch_size_per_im=512,
        fg_fraction=0.25,
        fg_thresh=0.5,
        bg_thresh_hi=0.5,
        bg_thresh_lo=0.0,
        bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
        class_nums=class_nums,
        use_random=use_random)
    rois.stop_gradient = True
    labels_int32.stop_gradient = True
    bbox_targets.stop_gradient = True
    bbox_inside_weights.stop_gradient = True
    bbox_outside_weights.stop_gradient = True

    pool5 = fluid.layers.roi_pool(
        input=res4,
        rois=rois,
        pooled_height=14,
        pooled_width=14,
        spatial_scale=0.0625)

    res5_2_sum = layer_warp(block_func, pool5, 512, stages[3], 2, name="res5")
    res5_pool = fluid.layers.pool2d(
        res5_2_sum, pool_type='avg', pool_size=7, name='res5_pool')
    cls_score = fluid.layers.fc(input=res5_pool,
                                size=class_nums,
                                act=None,
                                name='cls_score',
                                param_attr=ParamAttr(name='cls_score_w'),
                                bias_attr=ParamAttr(
                                    name='cls_score_b',
                                    learning_rate=2.,
                                    regularizer=L2Decay(0.)))
    bbox_pred = fluid.layers.fc(input=res5_pool,
                                size=4 * class_nums,
                                act=None,
                                name='bbox_pred',
                                param_attr=ParamAttr(name='bbox_pred_w'),
                                bias_attr=ParamAttr(
                                    name='bbox_pred_b',
                                    learning_rate=2.,
                                    regularizer=L2Decay(0.)))

    return rpn_cls_score, rpn_bbox_pred, anchor, var, cls_score, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois, labels_int32


def RPNloss(rpn_cls_prob, rpn_bbox_pred, anchor, var, gt_box, is_crowd, im_info,
            use_random):
    rpn_cls_score_reshape = fluid.layers.transpose(
        rpn_cls_prob, perm=[0, 2, 3, 1])
    rpn_bbox_pred_reshape = fluid.layers.transpose(
        rpn_bbox_pred, perm=[0, 2, 3, 1])
    anchor_reshape = fluid.layers.reshape(anchor, shape=(-1, 4))
    var_reshape = fluid.layers.reshape(var, shape=(-1, 4))
    #rpn_cls_score_reshape
    rpn_cls_score_reshape = fluid.layers.reshape(
        x=rpn_cls_score_reshape, shape=(0, -1, 1))
    #rpn_bbox_pred_reshape
    rpn_bbox_pred_reshape = fluid.layers.reshape(
        x=rpn_bbox_pred_reshape, shape=(0, -1, 4))
    score_pred, loc_pred, score_target, loc_target = fluid.layers.rpn_target_assign(
        bbox_pred=rpn_bbox_pred_reshape,
        cls_logits=rpn_cls_score_reshape,
        anchor_box=anchor_reshape,
        anchor_var=var_reshape,
        gt_boxes=gt_box,
        is_crowd=is_crowd,
        im_info=im_info,
        rpn_batch_size_per_im=256,
        rpn_straddle_thresh=0.0,
        rpn_fg_fraction=0.5,
        rpn_positive_overlap=0.7,
        rpn_negative_overlap=0.3,
        use_random=use_random)

    score_target = fluid.layers.cast(x=score_target, dtype='float32')
    rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
        x=score_pred, label=score_target)
    rpn_cls_loss = fluid.layers.reduce_mean(rpn_cls_loss, name='loss_rpn_cls')

    rpn_reg_loss = fluid.layers.smooth_l1(x=loc_pred, y=loc_target, sigma=3.0)
    rpn_reg_loss = fluid.layers.reduce_sum(rpn_reg_loss, name='loss_rpn_bbox')
    score_shape = fluid.layers.shape(score_target)
    score_shape = fluid.layers.cast(x=score_shape, dtype='float32')
    norm = fluid.layers.reduce_prod(score_shape)
    norm.stop_gradient = True
    rpn_reg_loss = rpn_reg_loss / norm
    #rpn_reg_loss.persistable = True
    #rpn_cls_loss.persistable = True

    return rpn_cls_loss, rpn_reg_loss
