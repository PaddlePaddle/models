import paddle.v2 as paddle
from config.pascal_voc_conf import cfg


def net_conf(mode):
    """Network configuration. Total three modes included 'train' 'eval'
    and 'infer'. Loss and mAP evaluation layer will return if using 'train'
    and 'eval'. In 'infer' mode, only detection output layer will be returned.
    """
    default_l2regularization = cfg.TRAIN.L2REGULARIZATION

    default_bias_attr = paddle.attr.ParamAttr(l2_rate=0.0, learning_rate=2.0)
    default_static_bias_attr = paddle.attr.ParamAttr(is_static=True)

    def get_param_attr(local_lr, regularization):
        is_static = False
        if local_lr == 0.0:
            is_static = True
        return paddle.attr.ParamAttr(
            learning_rate=local_lr, l2_rate=regularization, is_static=is_static)

    def get_loc_conf_filter_size(aspect_ratio_num, min_size_num, max_size_num):
        loc_filter_size = (
            aspect_ratio_num * 2 + min_size_num + max_size_num) * 4
        conf_filter_size = (
            aspect_ratio_num * 2 + min_size_num + max_size_num) * cfg.CLASS_NUM
        return loc_filter_size, conf_filter_size

    def conv_group(stack_num, name_list, input, filter_size_list, num_channels,
                   num_filters_list, stride_list, padding_list,
                   common_bias_attr, common_param_attr, common_act):
        conv = input
        in_channels = num_channels
        for i in xrange(stack_num):
            conv = paddle.layer.img_conv(
                name=name_list[i],
                input=conv,
                filter_size=filter_size_list[i],
                num_channels=in_channels,
                num_filters=num_filters_list[i],
                stride=stride_list[i],
                padding=padding_list[i],
                bias_attr=common_bias_attr,
                param_attr=common_param_attr,
                act=common_act)
            in_channels = num_filters_list[i]
        return conv

    def vgg_block(idx_str, input, num_channels, num_filters, pool_size,
                  pool_stride, pool_pad):
        layer_name = "conv%s_" % idx_str
        stack_num = 3
        name_list = [layer_name + str(i + 1) for i in xrange(3)]

        conv = conv_group(stack_num, name_list, input, [3] * stack_num,
                          num_channels, [num_filters] * stack_num,
                          [1] * stack_num, [1] * stack_num, default_bias_attr,
                          get_param_attr(1, default_l2regularization),
                          paddle.activation.Relu())

        pool = paddle.layer.img_pool(
            input=conv,
            pool_size=pool_size,
            num_channels=num_filters,
            pool_type=paddle.pooling.CudnnMax(),
            stride=pool_stride,
            padding=pool_pad)
        return conv, pool

    def mbox_block(layer_idx, input, num_channels, filter_size, loc_filters,
                   conf_filters):
        mbox_loc_name = layer_idx + "_mbox_loc"
        mbox_loc = paddle.layer.img_conv(
            name=mbox_loc_name,
            input=input,
            filter_size=filter_size,
            num_channels=num_channels,
            num_filters=loc_filters,
            stride=1,
            padding=1,
            bias_attr=default_bias_attr,
            param_attr=get_param_attr(1, default_l2regularization),
            act=paddle.activation.Identity())

        mbox_conf_name = layer_idx + "_mbox_conf"
        mbox_conf = paddle.layer.img_conv(
            name=mbox_conf_name,
            input=input,
            filter_size=filter_size,
            num_channels=num_channels,
            num_filters=conf_filters,
            stride=1,
            padding=1,
            bias_attr=default_bias_attr,
            param_attr=get_param_attr(1, default_l2regularization),
            act=paddle.activation.Identity())

        return mbox_loc, mbox_conf

    def ssd_block(layer_idx, input, img_shape, num_channels, num_filters1,
                  num_filters2, aspect_ratio, variance, min_size, max_size):
        layer_name = "conv" + layer_idx + "_"
        stack_num = 2
        conv1_name = layer_name + "1"
        conv2_name = layer_name + "2"
        conv2 = conv_group(stack_num, [conv1_name, conv2_name], input, [1, 3],
                           num_channels, [num_filters1, num_filters2], [1, 2],
                           [0, 1], default_bias_attr,
                           get_param_attr(1, default_l2regularization),
                           paddle.activation.Relu())

        loc_filters, conf_filters = get_loc_conf_filter_size(
            len(aspect_ratio), len(min_size), len(max_size))
        mbox_loc, mbox_conf = mbox_block(conv2_name, conv2, num_filters2, 3,
                                         loc_filters, conf_filters)
        mbox_priorbox = paddle.layer.priorbox(
            input=conv2,
            image=img_shape,
            min_size=min_size,
            max_size=max_size,
            aspect_ratio=aspect_ratio,
            variance=variance)

        return conv2, mbox_loc, mbox_conf, mbox_priorbox

    img = paddle.layer.data(
        name='image',
        type=paddle.data_type.dense_vector(cfg.IMG_CHANNEL * cfg.IMG_HEIGHT *
                                           cfg.IMG_WIDTH),
        height=cfg.IMG_HEIGHT,
        width=cfg.IMG_WIDTH)

    stack_num = 2
    conv1_2 = conv_group(stack_num, ['conv1_1', 'conv1_2'], img,
                         [3] * stack_num, 3, [64] * stack_num, [1] * stack_num,
                         [1] * stack_num, default_static_bias_attr,
                         get_param_attr(0, 0), paddle.activation.Relu())

    pool1 = paddle.layer.img_pool(
        name="pool1",
        input=conv1_2,
        pool_type=paddle.pooling.CudnnMax(),
        pool_size=2,
        num_channels=64,
        stride=2)

    stack_num = 2
    conv2_2 = conv_group(stack_num, ['conv2_1', 'conv2_2'], pool1, [3] *
                         stack_num, 64, [128] * stack_num, [1] * stack_num,
                         [1] * stack_num, default_static_bias_attr,
                         get_param_attr(0, 0), paddle.activation.Relu())

    pool2 = paddle.layer.img_pool(
        name="pool2",
        input=conv2_2,
        pool_type=paddle.pooling.CudnnMax(),
        pool_size=2,
        num_channels=128,
        stride=2)

    conv3_3, pool3 = vgg_block("3", pool2, 128, 256, 2, 2, 0)

    conv4_3, pool4 = vgg_block("4", pool3, 256, 512, 2, 2, 0)
    conv4_3_mbox_priorbox = paddle.layer.priorbox(
        input=conv4_3,
        image=img,
        min_size=cfg.NET.CONV4.PB.MIN_SIZE,
        aspect_ratio=cfg.NET.CONV4.PB.ASPECT_RATIO,
        variance=cfg.NET.CONV4.PB.VARIANCE)
    conv4_3_norm = paddle.layer.cross_channel_norm(
        name="conv4_3_norm",
        input=conv4_3,
        param_attr=paddle.attr.ParamAttr(
            initial_mean=20, initial_std=0, is_static=False, learning_rate=1))
    CONV4_PB = cfg.NET.CONV4.PB
    loc_filter_size, conf_filter_size = get_loc_conf_filter_size(
        len(CONV4_PB.ASPECT_RATIO),
        len(CONV4_PB.MIN_SIZE), len(CONV4_PB.MAX_SIZE))
    conv4_3_norm_mbox_loc, conv4_3_norm_mbox_conf = \
            mbox_block("conv4_3_norm", conv4_3_norm, 512, 3,
                    loc_filter_size, conf_filter_size)

    conv5_3, pool5 = vgg_block("5", pool4, 512, 512, 3, 1, 1)

    stack_num = 2
    fc7 = conv_group(stack_num, ['fc6', 'fc7'], pool5, [3, 1], 512, [1024] *
                     stack_num, [1] * stack_num, [1, 0], default_bias_attr,
                     get_param_attr(1, default_l2regularization),
                     paddle.activation.Relu())

    FC7_PB = cfg.NET.FC7.PB
    loc_filter_size, conf_filter_size = get_loc_conf_filter_size(
        len(FC7_PB.ASPECT_RATIO), len(FC7_PB.MIN_SIZE), len(FC7_PB.MAX_SIZE))
    fc7_mbox_loc, fc7_mbox_conf = mbox_block("fc7", fc7, 1024, 3,
                                             loc_filter_size, conf_filter_size)
    fc7_mbox_priorbox = paddle.layer.priorbox(
        input=fc7,
        image=img,
        min_size=cfg.NET.FC7.PB.MIN_SIZE,
        max_size=cfg.NET.FC7.PB.MAX_SIZE,
        aspect_ratio=cfg.NET.FC7.PB.ASPECT_RATIO,
        variance=cfg.NET.FC7.PB.VARIANCE)

    conv6_2, conv6_2_mbox_loc, conv6_2_mbox_conf, conv6_2_mbox_priorbox = \
            ssd_block("6", fc7, img, 1024, 256, 512,
                    cfg.NET.CONV6.PB.ASPECT_RATIO,
                    cfg.NET.CONV6.PB.VARIANCE,
                    cfg.NET.CONV6.PB.MIN_SIZE,
                    cfg.NET.CONV6.PB.MAX_SIZE)
    conv7_2, conv7_2_mbox_loc, conv7_2_mbox_conf, conv7_2_mbox_priorbox = \
            ssd_block("7", conv6_2, img, 512, 128, 256,
                    cfg.NET.CONV7.PB.ASPECT_RATIO,
                    cfg.NET.CONV7.PB.VARIANCE,
                    cfg.NET.CONV7.PB.MIN_SIZE,
                    cfg.NET.CONV7.PB.MAX_SIZE)
    conv8_2, conv8_2_mbox_loc, conv8_2_mbox_conf, conv8_2_mbox_priorbox = \
            ssd_block("8", conv7_2, img, 256, 128, 256,
                    cfg.NET.CONV8.PB.ASPECT_RATIO,
                    cfg.NET.CONV8.PB.VARIANCE,
                    cfg.NET.CONV8.PB.MIN_SIZE,
                    cfg.NET.CONV8.PB.MAX_SIZE)

    pool6 = paddle.layer.img_pool(
        name="pool6",
        input=conv8_2,
        pool_size=3,
        num_channels=256,
        stride=1,
        pool_type=paddle.pooling.Avg())
    POOL6_PB = cfg.NET.POOL6.PB
    loc_filter_size, conf_filter_size = get_loc_conf_filter_size(
        len(POOL6_PB.ASPECT_RATIO),
        len(POOL6_PB.MIN_SIZE), len(POOL6_PB.MAX_SIZE))
    pool6_mbox_loc, pool6_mbox_conf = mbox_block(
        "pool6", pool6, 256, 3, loc_filter_size, conf_filter_size)
    pool6_mbox_priorbox = paddle.layer.priorbox(
        input=pool6,
        image=img,
        min_size=cfg.NET.POOL6.PB.MIN_SIZE,
        max_size=cfg.NET.POOL6.PB.MAX_SIZE,
        aspect_ratio=cfg.NET.POOL6.PB.ASPECT_RATIO,
        variance=cfg.NET.POOL6.PB.VARIANCE)

    mbox_priorbox = paddle.layer.concat(
        name="mbox_priorbox",
        input=[
            conv4_3_mbox_priorbox, fc7_mbox_priorbox, conv6_2_mbox_priorbox,
            conv7_2_mbox_priorbox, conv8_2_mbox_priorbox, pool6_mbox_priorbox
        ])

    loc_loss_input = [
        conv4_3_norm_mbox_loc, fc7_mbox_loc, conv6_2_mbox_loc, conv7_2_mbox_loc,
        conv8_2_mbox_loc, pool6_mbox_loc
    ]

    conf_loss_input = [
        conv4_3_norm_mbox_conf, fc7_mbox_conf, conv6_2_mbox_conf,
        conv7_2_mbox_conf, conv8_2_mbox_conf, pool6_mbox_conf
    ]

    detection_out = paddle.layer.detection_output(
        input_loc=loc_loss_input,
        input_conf=conf_loss_input,
        priorbox=mbox_priorbox,
        confidence_threshold=cfg.NET.DETOUT.CONFIDENCE_THRESHOLD,
        nms_threshold=cfg.NET.DETOUT.NMS_THRESHOLD,
        num_classes=cfg.CLASS_NUM,
        nms_top_k=cfg.NET.DETOUT.NMS_TOP_K,
        keep_top_k=cfg.NET.DETOUT.KEEP_TOP_K,
        background_id=cfg.BACKGROUND_ID,
        name="detection_output")

    if mode == 'train' or mode == 'eval':
        bbox = paddle.layer.data(
            name='bbox', type=paddle.data_type.dense_vector_sequence(6))
        loss = paddle.layer.multibox_loss(
            input_loc=loc_loss_input,
            input_conf=conf_loss_input,
            priorbox=mbox_priorbox,
            label=bbox,
            num_classes=cfg.CLASS_NUM,
            overlap_threshold=cfg.NET.MBLOSS.OVERLAP_THRESHOLD,
            neg_pos_ratio=cfg.NET.MBLOSS.NEG_POS_RATIO,
            neg_overlap=cfg.NET.MBLOSS.NEG_OVERLAP,
            background_id=cfg.BACKGROUND_ID,
            name="multibox_loss")
        paddle.evaluator.detection_map(
            input=detection_out,
            label=bbox,
            overlap_threshold=cfg.NET.DETMAP.OVERLAP_THRESHOLD,
            background_id=cfg.BACKGROUND_ID,
            evaluate_difficult=cfg.NET.DETMAP.EVAL_DIFFICULT,
            ap_type=cfg.NET.DETMAP.AP_TYPE,
            name="detection_evaluator")
        return loss, detection_out
    elif mode == 'infer':
        return detection_out
