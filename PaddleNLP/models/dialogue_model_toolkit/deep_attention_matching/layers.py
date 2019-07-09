"""
Layers
"""

import paddle.fluid as fluid


def loss(x, y, clip_value=10.0):
    """Calculate the sigmoid cross entropy with logits for input(x).

    Args:
        x: Variable with shape with shape [batch, dim]
        y: Input label

    Returns:
        loss: cross entropy
        logits: prediction
    """

    logits = fluid.layers.fc(
        input=x,
        size=1,
        bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(0.)))
    loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=logits, label=y)
    loss = fluid.layers.reduce_mean(
        fluid.layers.clip(
            loss, min=-clip_value, max=clip_value))
    return loss, logits


def ffn(input, d_inner_hid, d_hid, name=None):
    """Position-wise Feed-Forward Network
    """

    hidden = fluid.layers.fc(input=input,
                             size=d_inner_hid,
                             num_flatten_dims=2,
                             param_attr=fluid.ParamAttr(name=name + '_fc.w_0'),
                             bias_attr=fluid.ParamAttr(
                                 name=name + '_fc.b_0',
                                 initializer=fluid.initializer.Constant(0.)),
                             act="relu")
    out = fluid.layers.fc(input=hidden,
                          size=d_hid,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(name=name + '_fc.w_1'),
                          bias_attr=fluid.ParamAttr(
                              name=name + '_fc.b_1',
                              initializer=fluid.initializer.Constant(0.)))
    return out


def dot_product_attention(query,
                          key,
                          value,
                          d_key,
                          q_mask=None,
                          k_mask=None,
                          dropout_rate=None,
                          mask_cache=None):
    """Dot product layer.

     Args:
         query: a tensor with shape [batch, Q_time, Q_dimension]
         key: a tensor with shape [batch, time, K_dimension]
         value: a tensor with shape [batch, time, V_dimension]

         q_lengths: a tensor with shape [batch]
         k_lengths: a tensor with shape [batch]

     Returns:
         a tensor with shape [batch, query_time, value_dimension]

     Raises:
         AssertionError: if Q_dimension not equal to K_dimension when attention 
                        type is dot.
    """

    logits = fluid.layers.matmul(
        x=query, y=key, transpose_y=True, alpha=d_key ** (-0.5))

    if (q_mask is not None) and (k_mask is not None):
        if mask_cache is not None and q_mask.name in mask_cache and k_mask.name in mask_cache[
                q_mask.name]:
            mask, another_mask = mask_cache[q_mask.name][k_mask.name]
        else:
            mask = fluid.layers.matmul(x=q_mask, y=k_mask, transpose_y=True)
            another_mask = fluid.layers.scale(
                mask,
                scale=float(2 ** 32 - 1),
                bias=float(-1),
                bias_after_scale=False)
            if mask_cache is not None:
                if q_mask.name not in mask_cache:
                    mask_cache[q_mask.name] = dict()

                mask_cache[q_mask.name][k_mask.name] = [mask, another_mask]

        logits = mask * logits + another_mask

    attention = fluid.layers.softmax(logits)
    if dropout_rate:
        attention = fluid.layers.dropout(
            input=attention, dropout_prob=dropout_rate, is_test=False, seed=2)

    atten_out = fluid.layers.matmul(x=attention, y=value)

    return atten_out


def block(name,
          query,
          key,
          value,
          d_key,
          q_mask=None,
          k_mask=None,
          is_layer_norm=True,
          dropout_rate=None,
          mask_cache=None):
    """
    Block
    """
    att_out = dot_product_attention(
        query,
        key,
        value,
        d_key,
        q_mask,
        k_mask,
        dropout_rate,
        mask_cache=mask_cache)

    y = query + att_out
    if is_layer_norm:
        y = fluid.layers.layer_norm(
            input=y,
            begin_norm_axis=len(y.shape) - 1,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(1.),
                name=name + '_layer_norm.w_0'),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.),
                name=name + '_layer_norm.b_0'))

    z = ffn(y, d_key, d_key, name)
    w = y + z
    if is_layer_norm:
        w = fluid.layers.layer_norm(
            input=w,
            begin_norm_axis=len(w.shape) - 1,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(1.),
                name=name + '_layer_norm.w_1'),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.),
                name=name + '_layer_norm.b_1'))

    return w


def cnn_3d(input, out_channels_0, out_channels_1, add_relu=True):
    """
    CNN-3d
    """
    # same padding
    conv_0 = fluid.layers.conv3d(
        name="conv3d_0",
        input=input,
        num_filters=out_channels_0,
        filter_size=[3, 3, 3],
        padding=[1, 1, 1],
        act="elu" if add_relu else None,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
            low=-0.01, high=0.01)),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.0)))

    # same padding 
    pooling_0 = fluid.layers.pool3d(
        input=conv_0,
        pool_type="max",
        pool_size=3,
        pool_padding=1,
        pool_stride=3)

    conv_1 = fluid.layers.conv3d(
        name="conv3d_1",
        input=pooling_0,
        num_filters=out_channels_1,
        filter_size=[3, 3, 3],
        padding=[1, 1, 1],
        act="elu" if add_relu else None,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
            low=-0.01, high=0.01)),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.0)))

    # same padding 
    pooling_1 = fluid.layers.pool3d(
        input=conv_1,
        pool_type="max",
        pool_size=3,
        pool_padding=1,
        pool_stride=3)

    return pooling_1
