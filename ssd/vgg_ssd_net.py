import paddle.v2 as paddle
from config.pascal_voc_conf import cfg


def net_conf(mode):
    """Network configuration. Total three modes included 'train' 'eval'
    and 'infer'. Loss and mAP evaluation layer will return if using 'train'
    and 'eval'. In 'infer' mode, only detection output layer will be returned.
    """
    default_l2regularization = cfg.TRAIN.L2REGULARIZATION

    default_bias_attr = paddle.attr.ParamAttr(
        l2_rate=0.0, learning_rate=2.0, momentum=cfg.TRAIN.MOMENTUM)
    default_static_bias_attr = paddle.attr.ParamAttr(is_static=True)

    def xavier(channels, filter_size, local_lr, regularization):
        init_w = (3.0 / (filter_size**2 * channels))**0.5
        is_static = False
        if local_lr == 0.0:
            is_static = True
        return paddle.attr.ParamAttr(
            initial_min=(0.0 - init_w),
            initial_max=init_w,
            learning_rate=local_lr,
            l2_rate=regularization,
            momentum=cfg.TRAIN.MOMENTUM,
            is_static=is_static)

    def vgg_block(idx_str, input, num_channels, num_filters, pool_size,
                  pool_stride, pool_pad):
        layer_name = "conv%s_" % idx_str
        conv1 = paddle.layer.img_conv(
            name=layer_name + "1",
            input=input,
            filter_size=3,
            num_channels=num_channels,
            num_filters=num_filters,
            stride=1,
            padding=1,
            bias_attr=default_bias_attr,
            param_attr=xavier(num_filters, 3, 1, default_l2regularization),
            act=paddle.activation.Relu())
        conv2 = paddle.layer.img_conv(
            name=layer_name + "2",
            input=conv1,
            filter_size=3,
            num_channels=num_filters,
            num_filters=num_filters,
            stride=1,
            padding=1,
            bias_attr=default_bias_attr,
            param_attr=xavier(num_filters, 3, 1, default_l2regularization),
            act=paddle.activation.Relu())
        conv3 = paddle.layer.img_conv(
            name=layer_name + "3",
            input=conv2,
            filter_size=3,
            num_channels=num_filters,
            num_filters=num_filters,
            stride=1,
            padding=1,
            bias_attr=default_bias_attr,
            param_attr=xavier(num_filters, 3, 1, default_l2regularization),
            act=paddle.activation.Relu())
        pool = paddle.layer.img_pool(
            input=conv3,
            pool_size=pool_size,
            num_channels=num_filters,
            pool_type=paddle.pooling.CudnnMax(),
            stride=pool_stride,
            padding=pool_pad)
        return conv3, pool

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
            param_attr=xavier(loc_filters, filter_size, 1,
                              default_l2regularization),
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
            param_attr=xavier(conf_filters, filter_size, 1,
                              default_l2regularization),
            act=paddle.activation.Identity())

        return mbox_loc, mbox_conf

    def ssd_block(layer_idx, input, img_shape, num_channels, num_filters1,
                  num_filters2, aspect_ratio, variance, min_size, max_size):
        layer_name = "conv" + layer_idx + "_"
        conv1_name = layer_name + "1"
        conv1 = paddle.layer.img_conv(
            name=conv1_name,
            input=input,
            filter_size=1,
            num_channels=num_channels,
            num_filters=num_filters1,
            stride=1,
            padding=0,
            bias_attr=default_bias_attr,
            param_attr=xavier(num_filters1, 1, 1, default_l2regularization),
            act=paddle.activation.Relu())
        conv2_name = layer_name + "2"
        conv2 = paddle.layer.img_conv(
            name=conv2_name,
            input=conv1,
            filter_size=3,
            num_channels=num_filters1,
            num_filters=num_filters2,
            stride=2,
            padding=1,
            bias_attr=default_bias_attr,
            param_attr=xavier(num_filters2, 3, 1, default_l2regularization),
            act=paddle.activation.Relu())

        loc_filters = (len(aspect_ratio) * 2 + 1 + len(max_size)) * 4
        conf_filters = (
            len(aspect_ratio) * 2 + 1 + len(max_size)) * cfg.CLASS_NUM
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

    conv1_1 = paddle.layer.img_conv(
        name="conv1_1",
        input=img,
        filter_size=3,
        num_channels=3,
        num_filters=64,
        stride=1,
        padding=1,
        bias_attr=default_static_bias_attr,
        param_attr=xavier(64, 3, 0, 0),
        act=paddle.activation.Relu())
    conv1_2 = paddle.layer.img_conv(
        name="conv1_2",
        input=conv1_1,
        filter_size=3,
        num_channels=64,
        num_filters=64,
        stride=1,
        padding=1,
        bias_attr=default_static_bias_attr,
        param_attr=xavier(64, 3, 0, 0),
        act=paddle.activation.Relu())
    pool1 = paddle.layer.img_pool(
        name="pool1",
        input=conv1_2,
        pool_type=paddle.pooling.CudnnMax(),
        pool_size=2,
        num_channels=64,
        stride=2)

    conv2_1 = paddle.layer.img_conv(
        name="conv2_1",
        input=pool1,
        filter_size=3,
        num_channels=64,
        num_filters=128,
        stride=1,
        padding=1,
        bias_attr=default_static_bias_attr,
        param_attr=xavier(128, 3, 0, 0),
        act=paddle.activation.Relu())
    conv2_2 = paddle.layer.img_conv(
        name="conv2_2",
        input=conv2_1,
        filter_size=3,
        num_channels=128,
        num_filters=128,
        stride=1,
        padding=1,
        bias_attr=default_static_bias_attr,
        param_attr=xavier(128, 3, 0, 0),
        act=paddle.activation.Relu())
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
            initial_mean=20,
            initial_std=0,
            is_static=False,
            learning_rate=1,
            momentum=cfg.TRAIN.MOMENTUM))
    conv4_3_norm_mbox_loc, conv4_3_norm_mbox_conf = \
            mbox_block("conv4_3_norm", conv4_3_norm, 512, 3, 12, 63)

    conv5_3, pool5 = vgg_block("5", pool4, 512, 512, 3, 1, 1)

    fc6 = paddle.layer.img_conv(
        name="fc6",
        input=pool5,
        filter_size=3,
        num_channels=512,
        num_filters=1024,
        stride=1,
        padding=1,
        bias_attr=default_bias_attr,
        param_attr=xavier(1024, 3, 1, default_l2regularization),
        act=paddle.activation.Relu())

    fc7 = paddle.layer.img_conv(
        name="fc7",
        input=fc6,
        filter_size=1,
        num_channels=1024,
        num_filters=1024,
        stride=1,
        padding=0,
        bias_attr=default_bias_attr,
        param_attr=xavier(1024, 1, 1, default_l2regularization),
        act=paddle.activation.Relu())
    fc7_mbox_loc, fc7_mbox_conf = mbox_block("fc7", fc7, 1024, 3, 24, 126)
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
    pool6_mbox_loc, pool6_mbox_conf = mbox_block("pool6", pool6, 256, 3, 24,
                                                 126)
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
