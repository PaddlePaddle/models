from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid


def conv_bn_layer(input, ch_out, filter_size, stride, padding, act='relu'):
    parameter_attr = ParamAttr(learning_rate=0.01)
    conv1 = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=parameter_attr,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv1, act=act)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]  # if args.data_format == 'NCHW' else input.shape[-1]
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_out, stride):
    short = shortcut(input, ch_out, stride)
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def bottleneck(input, ch_out, stride):
    short = shortcut(input, ch_out * 4, stride)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1)
    conv3 = conv_bn_layer(conv2, ch_out * 4, 1, 1, 0, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv3, act='relu')


def layer_warp(block_func, input, ch_out, count, stride):
    res_out = block_func(input, ch_out, stride)
    for i in range(1, count):
        res_out = block_func(res_out, ch_out, 1)
    return res_out


def FasterRcnn(input, depth, anchor_sizes, variance, aspect_ratios, gt_box,
               gt_label, im_info, num_classes):

    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]

    conv1 = conv_bn_layer(input, ch_out=64, filter_size=7, stride=2, padding=3)

    pool1 = fluid.layers.pool2d(
        input=conv1, pool_type='max', pool_size=3, pool_stride=2)

    res2 = layer_warp(block_func, pool1, 64, stages[0], 1)
    res3 = layer_warp(block_func, res2, 128, stages[1], 2)
    res4 = layer_warp(block_func, res3, 256, stages[2], 2)

    #========= RPN ============

    # rpn_conv/3*3 num_filter = 256/512

    rpn_conv = fluid.layers.conv2d(
        input=res4,
        num_filters=1024,
        filter_size=3,
        stride=1,
        padding=1,
        act='relu')

    anchor, var = fluid.layers.anchor_generator(
        input=rpn_conv,
        anchor_sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
        variance=variance,
        stride=[16.0, 16.0])

    num_anchor = anchor.shape[2]
    #rpn_cls_score
    rpn_cls_score = fluid.layers.conv2d(
        rpn_conv,
        num_filters=num_anchor,
        filter_size=1,
        stride=1,
        padding=0,
        act='relu')

    #rpn_bbox_pred
    rpn_bbox_pred = fluid.layers.conv2d(
        rpn_conv,
        num_filters=4 * num_anchor,
        filter_size=1,
        stride=1,
        padding=0,
        act='relu')

    rpn_cls_score_prob = fluid.layers.sigmoid(rpn_cls_score)
    rpn_cls_score_prob = fluid.layers.transpose(
        rpn_cls_score_prob, perm=[0, 3, 1, 2])

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
        gt_boxes=gt_box,
        im_scales=im_info[:, 2],
        batch_size_per_im=256,
        fg_fraction=0.25,
        fg_thresh=0.25,
        bg_thresh_hi=0.5,
        bg_thresh_lo=0.0,
        bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
        class_nums=class_nums)

    pool5 = fluid.layers.roi_pool(
        input=res4,
        rois=rois,
        pooled_height=14,
        pooled_width=14,
        spatial_scale=0.0625)

    res5_0_branch2a = fluid.layers.conv2d(
        pool5, num_filters=512, filter_size=1, stride=2, padding=0, act=None)

    res5_2_sum = layer_warp(block_func, res5_0_branch2a, 512, stages[3], 2)
    res5_pool = fluid.layers.pool2d(res5_2_sum, pool_type='avg', pool_size=7)

    cls_score = fluid.layers.fc(input=res5_pool, size=class_nums, act='relu')
    bbox_pred = fluid.layers.fc(input=res5_pool,
                                size=4 * class_nums,
                                act='relu')

    return rpn_cls_score, rpn_bbox_pred, anchor, var, cls_score, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois, labels_int32


def RPNloss(rpn_cls_prob, rpn_bbox_pred, anchor, var, gt_box):
    rpn_cls_score_reshape = fluid.layers.transpose(
        rpn_cls_prob, perm=[0, 2, 3, 1])
    rpn_bbox_pred_reshape = fluid.layers.transpose(
        rpn_bbox_pred, perm=[0, 2, 3, 1])

    #rpn_cls_score_reshape
    rpn_cls_score_reshape = fluid.layers.reshape(
        x=rpn_cls_score_reshape, shape=(0, -1, 1))

    #rpn_bbox_pred_reshape
    rpn_bbox_pred_reshape = fluid.layers.reshape(
        x=rpn_bbox_pred_reshape, shape=(0, -1, 4))

    score_pred, loc_pred, score_target, loc_target = fluid.layers.rpn_target_assign(
        loc=rpn_bbox_pred_reshape,
        scores=rpn_cls_score_reshape,
        anchor_box=anchor,
        anchor_var=var,
        gt_box=gt_box,
        rpn_batch_size_per_im=512,
        fg_fraction=0.25,
        rpn_positive_overlap=0.7,
        rpn_negative_overlap=0.3)

    cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
        logits=score_pred, label=score_target)
    cls_loss = fluid.layers.reduce_sum(cls_loss)

    #num_foreground = loc_target.shape[0]
    #inside_weight = fluid.layers.fill_constant(shape=[num_foreground],value=1.0)
    #outside_weight = fluid.layers.fill_constant(shape=[num_foreground],value=1.0/num_foreground)
    reg_loss = fluid.layers.smooth_l1(x=loc_pred, y=loc_target, sigma=3.0)
    reg_loss = fluid.layers.reduce_sum(reg_loss)

    return cls_loss, reg_loss
