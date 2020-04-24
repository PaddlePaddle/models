import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper


def dsq_round(x):
    delta = np.max(x) - np.min(x)
    x = (x / delta + 0.5)
    return x.round() * 2 - 1


def dsq_round_back(dy):
    return np.array(dy)


def dsq(x, bit=8, name=None):
    def clip(x, upper, lower):
        x = x + fluid.layers.relu(lower - x)
        x = x - fluid.layers.relu(x - upper)
        return x

    def phi_function(x, mi, alpha, delta):
        s = 1 / (1 - alpha)
        k = fluid.layers.log(2 / alpha - 1) * (1 / delta)
        res = (fluid.layers.tanh((x - mi) * k)) * s
        return res

    def dequantize(x, lower_bound, delta, interval):

        # save mem
        res = ((x + 1) / 2 + interval) * delta + lower_bound
        return res

    helper = LayerHelper("dsq", **locals())
    dtype = 'float32'
    bit_range = 2**bit - 1

    u_param_attr = fluid.ParamAttr(
        initializer=fluid.initializer.ConstantInitializer(value=(2**31 - 1)))
    l_param_attr = fluid.ParamAttr(
        initializer=fluid.initializer.ConstantInitializer(value=(-1) *
                                                          (2**31 - 1)))

    alpha_param_attr = fluid.ParamAttr(
        initializer=fluid.initializer.ConstantInitializer(value=0.2))
    u_param = helper.create_parameter(attr=u_param_attr, shape=[1], dtype=dtype)
    l_param = helper.create_parameter(attr=l_param_attr, shape=[1], dtype=dtype)
    alpha_param = helper.create_parameter(
        attr=alpha_param_attr, shape=[1], dtype=dtype)

    upper = fluid.layers.create_global_var(
        shape=[1], value=(2**31 - 1), dtype='float32', persistable=True)
    lower = fluid.layers.create_global_var(
        shape=[1], value=(-1) * (2**31 - 1), dtype='float32', persistable=True)
    fluid.layers.assign(upper * 0.9 + u_param * 0.1, upper)
    fluid.layers.assign(lower * 0.9 + l_param * 0.1, lower)
    x = clip(x, upper, lower)
    cur_max = fluid.layers.reduce_max(x)
    cur_min = fluid.layers.reduce_min(x)
    delta = (cur_max - cur_min) / bit_range
    interval = (x - cur_min) / delta
    interval = fluid.layers.floor(interval)
    mi = (interval + 0.5) * delta + cur_min
    phi_x = phi_function(x, mi, alpha_param, delta)
    out_var = fluid.default_main_program().current_block().create_var(
        dtype=dtype, shape=phi_x.shape)
    fluid.layers.py_func(
        func=dsq_round,
        x=phi_x,
        out=out_var,
        backward_func=dsq_round_back,
        skip_vars_in_backward_input=[phi_x, out_var])
    x = dequantize(out_var, cur_min, delta, interval)

    return x, delta, x


def pact(x, name=None):
    helper = LayerHelper("pact", **locals())
    dtype = 'float32'
    u_param_attr = fluid.ParamAttr(
        name=x.name + '_pact',
        initializer=fluid.initializer.ConstantInitializer(value=15),
        regularizer=fluid.regularizer.L2Decay(0.1),
        learning_rate=100)
    u_param = helper.create_parameter(attr=u_param_attr, shape=[1], dtype=dtype)
    x = x - fluid.layers.relu(x - u_param)
    #x = fluid.layers.relu(x)
    return x
