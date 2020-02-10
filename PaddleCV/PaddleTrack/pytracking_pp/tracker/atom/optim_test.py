import torch
import torch.nn.functional as F
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import layers
from bilib import crash_on_ipy

import os.path as osp
import sys

from pytracking_pp.libs.paddle_utils import n2t, p2n, t2n, leaky_relu, elu

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..', '..'))

from pytracking_pp.libs import TensorList
from pytracking.libs import TensorList as TensorListT

import pytracking_pp.libs.optimization as optim_static
import pytracking.libs.optimization as optim_t
from pytracking_pp.tracker.atom.optim import ConvProblem, FactorizedConvProblem
from pytracking.tracker.atom.optim import ConvProblem as ConvProblemT
from pytracking.tracker.atom.optim import FactorizedConvProblem as FactorizedConvProblemT


def test_ConvProblem_forward_backward():
    rng = np.random.RandomState(0)
    ts = [rng.uniform(-1, 1, [32, 3, 3]).astype('float32') for _ in range(10)]
    ys = [rng.uniform(-1, 1, [1, 3, 3]).astype('float32') for _ in range(10)]
    sw = [rng.uniform(0, 1, [1]).astype('float32') for _ in range(10)]
    reg = rng.uniform(0, 1)
    var = rng.uniform(-1, 1, [1, 32, 3, 3]).astype('float32')

    # Paddle
    training_samples = TensorList([ts])
    y = TensorList([ys])
    filter_reg = TensorList([reg])
    sample_weights = TensorList([sw])
    variable = TensorList([var])

    act_param = 0.05
    mlu = lambda x: layers.elu(leaky_relu(x, 1 / act_param), act_param)
    problem = ConvProblem(training_samples, y, filter_reg, sample_weights,
                          response_activation=mlu)

    # Construct graph
    train_program = fluid.Program()
    start_program = fluid.Program()
    with fluid.program_guard(train_program, start_program):
        scope = 'first/'
        x_ph = TensorList(
            [fluid.layers.data('{}x_{}'.format(scope, idx), v.shape,
                               append_batch_size=False,
                               stop_gradient=False)
             for idx, v in enumerate(variable)])

        f0 = problem(x_ph, scope)
        loss = problem.ip_output(f0, f0)
        grad = TensorList(fluid.gradients(loss, x_ph))

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(program=fluid.default_startup_program())
    compiled_prog = fluid.compiler.CompiledProgram(train_program)

    scope = 'first/'
    feed_dict = problem.get_feed_dict(scope)
    # add variable feed
    for idx, v in enumerate(variable):
        feed_dict['{}x_{}'.format(scope, idx)] = v
    res = exe.run(compiled_prog, feed=feed_dict, fetch_list=[v.name for v in f0] + [v.name for v in grad])
    res_paddle = TensorList(res)

    # PyTorch
    training_samples = TensorListT([n2t(np.stack(ts))])
    y = TensorListT([n2t(ys)])
    filter_reg = TensorListT([n2t(reg)])
    sample_weights = TensorListT([n2t(np.stack(sw))])

    variable = TensorListT([n2t(var)])
    variable.requires_grad_(True)

    mlu = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
    problem = ConvProblemT(training_samples, y, filter_reg, sample_weights,
                           response_activation=mlu)
    f0 = problem(variable)
    loss = problem.ip_output(f0, f0)
    grad = TensorListT(torch.autograd.grad(loss, variable)).detach().cpu().numpy()
    res_torch = f0.detach().cpu().numpy()
    res_torch.extend(grad)

    for a, b in zip(res_paddle, res_torch):
        np.testing.assert_allclose(a, b, rtol=1e-04, atol=1e-05)


def test_FactorizedConvProblem_forward_backward():
    rng = np.random.RandomState(0)
    ts = [rng.uniform(-1, 1, [32, 3, 3]).astype('float32') for _ in range(10)]
    ys = [rng.uniform(-1, 1, [1, 3, 3]).astype('float32') for _ in range(10)]
    sw = [rng.uniform(0, 1, [1]).astype('float32') for _ in range(10)]
    reg = rng.uniform(0, 1)
    reg2 = rng.uniform(0, 1)
    var1 = rng.uniform(-1, 1, [1, 16, 3, 3]).astype('float32')
    var2 = rng.uniform(-1, 1, [16, 32, 1, 1]).astype('float32')

    # Paddle
    training_samples = TensorList([ts])
    y = TensorList([ys])
    filter_reg = TensorList([reg])
    project_reg = TensorList([reg2])
    sample_weights = TensorList([sw])
    variable = TensorList([var1.copy(), var2.copy()])

    act_param = 0.05
    mlu = lambda x: layers.elu(leaky_relu(x, 1 / act_param), act_param)
    problem = FactorizedConvProblem(training_samples, y,
                                    filter_reg=filter_reg,
                                    projection_reg=project_reg,
                                    params=None,
                                    sample_weights=sample_weights,
                                    projection_activation=lambda x: x,
                                    response_activation=mlu)

    # Construct graph
    train_program = fluid.Program()
    start_program = fluid.Program()
    with fluid.program_guard(train_program, start_program):
        scope = 'first/'
        x_ph = TensorList(
            [fluid.layers.data('{}x_{}'.format(scope, idx), v.shape,
                               append_batch_size=False,
                               stop_gradient=False)
             for idx, v in enumerate(variable)])

        f0 = problem(x_ph, scope)
        loss = problem.ip_output(f0, f0)
        grad = TensorList(fluid.gradients(loss, x_ph))

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(program=fluid.default_startup_program())
    compiled_prog = fluid.compiler.CompiledProgram(train_program)

    scope = 'first/'
    feed_dict = problem.get_feed_dict(scope)
    # add variable feed
    for idx, v in enumerate(variable):
        feed_dict['{}x_{}'.format(scope, idx)] = v
    res = exe.run(compiled_prog, feed=feed_dict, fetch_list=[v.name for v in f0] + [v.name for v in grad])
    res_paddle = TensorList(res)

    # PyTorch
    training_samples = TensorListT([n2t(np.stack(ts))])
    y = TensorListT([n2t(ys)])
    filter_reg = TensorListT([n2t(reg)])
    project_reg = TensorListT([n2t(reg2)])
    sample_weights = TensorListT([n2t(np.stack(sw))])

    variable = TensorListT([n2t(var1.copy()), n2t(var2.copy())])
    variable.requires_grad_(True)

    mlu = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
    problem = FactorizedConvProblemT(training_samples, y,
                                     filter_reg=filter_reg,
                                     projection_reg=project_reg,
                                     params=None,
                                     sample_weights=sample_weights,
                                     projection_activation=lambda x: x,
                                     response_activation=mlu)
    f0 = problem(variable)
    loss = problem.ip_output(f0, f0)
    grad = TensorListT(torch.autograd.grad(loss, variable)).detach().cpu().numpy()
    res_torch = f0.detach().cpu().numpy()
    res_torch.extend(grad)

    for a, b in zip(res_paddle, res_torch):
        np.testing.assert_allclose(a, b, rtol=1e-04, atol=1e-05)


if __name__ == '__main__':
    # test_ConvProblem_forward_backward()
    test_FactorizedConvProblem_forward_backward()