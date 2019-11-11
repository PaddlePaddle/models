import numpy as np

# asymmetric gaussian shaped activation function g_A 
import paddle.fluid as fluid


def GaussActivation(input, a, mu, sigma1, sigma2):
    initializer = fluid.initializer.ConstantInitializer(value=a)
    a = fluid.layers.create_parameter(
        shape=[1], dtype='float32', default_initializer=initializer)
    a = fluid.layers.clip(a, min=1.01, max=6.0)

    initializer = fluid.initializer.ConstantInitializer(value=mu)
    mu = fluid.layers.create_parameter(
        shape=[1], dtype='float32', default_initializer=initializer)
    mu = fluid.layers.clip(mu, min=0.1, max=3.0)

    initializer = fluid.initializer.ConstantInitializer(value=sigma1)
    sigma1 = fluid.layers.create_parameter(
        shape=[1], dtype='float32', default_initializer=initializer)
    sigma1 = fluid.layers.clip(sigma1, min=1.0, max=2.0)

    initializer = fluid.initializer.ConstantInitializer(value=sigma2)
    sigma2 = fluid.layers.create_parameter(
        shape=[1], dtype='float32', default_initializer=initializer)
    sigma2 = fluid.layers.clip(sigma2, min=1.0, max=2.0)

    lowerThanMu = fluid.layers.less_than(input, mu)
    largerThanMu = fluid.layers.logical_not(lowerThanMu)

    diff_mu = (input - mu)
    leftValuesActiv = fluid.layers.exp(-1 * fluid.layers.square(diff_mu) *
                                       sigma1) * a
    leftValuesActiv = leftValuesActiv * lowerThanMu

    rightValueActiv = 1 + fluid.layers.exp(-1 * fluid.layers.square(diff_mu) *
                                           sigma2) * (a - 1)
    rightValueActiv = rightValueActiv * largerThanMu

    output = leftValuesActiv + rightValueActiv

    return output


def MaskUpdate(input, alpha):
    initializer = fluid.initializer.ConstantInitializer(value=alpha)
    alpha_t = fluid.layers.create_parameter(
        shape=[1], dtype='float32', default_initializer=initializer)

    alpha_t = fluid.layers.clip(alpha_t, min=0.6, max=0.8)
    out = fluid.layers.relu(input)
    out = fluid.layers.elementwise_pow(out, alpha_t)
    return out
