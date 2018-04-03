import paddle.v2 as paddle


__all__ = ['squeezenet']


def fire_module(x, chs, squeeze=16, expand=64):
    squeezer = paddle.layer.img_conv(
        input=x,
        num_channels=chs,
        filter_size=(1,1),
        num_filters=squeeze,
        stride=1,
        padding=(0,0),
        act=paddle.activation.Relu(),
        bias_attr=False)

    uno_expander = paddle.layer.img_conv(
        input=squeezer,
        filter_size=(1,1),
        num_filters=squeeze,
        stride=1,
        padding=(0,0),
        act=paddle.activation.Relu(),
        bias_attr=False)
    
    tri_expander = paddle.layer.img_conv(
        input=squeezer,
        filter_size=(3,3),
        num_filters=squeeze,
        stride=1,
        padding=(1,1),
        act=paddle.activation.Relu(),
        bias_attr=False)
    
    return paddle.layer.concat(input=[uno_expander, tri_expander])

def squeezenet(x, class_dim, include_top=True):
    conv1 = paddle.layer.img_conv(
        input=x,
        num_channels=3,
        filter_size=(3,3),
        num_filters=64,
        stride=(2,2),
        padding=(0,0),
        act=paddle.activation.Relu())
    pool1 = paddle.layer.img_pool(input=conv1, pool_size=3, stride=2, pool_type=paddle.pooling.Max())
    
    
    f1 = fire_module(pool1, 64, squeeze=16, expand=64)
    f2 = fire_module(f1, 32, squeeze=16, expand=64)
    pool2 = paddle.layer.img_pool(input=f2, num_channels=32, pool_size=3, stride=2, pool_type=paddle.pooling.Max())
    
    
    f3 = fire_module(pool2, 32, squeeze=32, expand=128)
    f4 = fire_module(f3, 64, squeeze=32, expand=128)
    pool3 = paddle.layer.img_pool(input=f4, num_channels=64, pool_size=3, stride=2, pool_type=paddle.pooling.Max())

    f5 = fire_module(pool3, 64, squeeze=48, expand=192)
    f6 = fire_module(f5, 96, squeeze=48, expand=192)
    f7 = fire_module(f6, 96, squeeze=64, expand=256)
    f8 = fire_module(f7, 128, squeeze=64, expand=256)
    
    if include_top:
        drop = paddle.layer.dropout(input=f8, dropout_rate=0.5)
        finalconv = paddle.layer.img_conv(
            input=drop,
            num_channels=128,
            filter_size=(1,1),
            num_filters=class_dim,
            stride=1,
            padding=(0,0),
            act=paddle.activation.Relu(),
            bias_attr=False)
        ### TODO: I'm trying to implement a global average pooling layer here.
        ### When I was using this layer, I manually set the pool_size to match the
        ### input dimensions. I saw that PaddleFluid has global pooling and wasn't
        ### sure what normal Paddle's equivalent is.
        gavg = paddle.layer.img_pool(input=finalconv, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())
        out = paddle.layer.fc(input=finalconv,
                          size=class_dim,
                          act=paddle.activation.Softmax())
    else:
        ### TODO: I'm trying to implement a global average pooling layer here.
        ### When I was using this layer, I manually set the pool_size to match the
        ### input dimensions. I saw that PaddleFluid has global pooling and wasn't
        ### sure what normal Paddle's equivalent is.
        gavg = paddle.layer.img_pool(input=f8, num_channels=128, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())
        out = paddle.layer.fc(input=f8,
                          size=class_dim,
                          act=paddle.activation.Softmax())
    return out

