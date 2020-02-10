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
sys.path.append(osp.join(CURRENT_DIR, '..', '..'))

from pytracking_pp.libs import TensorList
from pytracking.libs import TensorList as TensorListT

import pytracking_pp.libs.optimization as optim_static
import pytracking.libs.optimization as optim_t
from pytracking_pp.tracker.atom.optim import ConvProblem, FactorizedConvProblem
from pytracking.tracker.atom.optim import ConvProblem as ConvProblemT
from pytracking.tracker.atom.optim import FactorizedConvProblem as FactorizedConvProblemT


def test_ConvProblem_gradient_descent():
    rng = np.random.RandomState(0)
    ts = [rng.uniform(-1, 1, [32, 3, 3]).astype('float32') for _ in range(10)]
    ys = [rng.uniform(-1, 1, [1, 3, 3]).astype('float32') for _ in range(10)]
    sw = [rng.uniform(0, 1, [1]).astype('float32') for _ in range(10)]
    reg = rng.uniform(0, 1)
    var = rng.uniform(-1, 1, [1, 32, 3, 3]).astype('float32')

    # Paddle run
    training_samples = TensorList([ts])
    y = TensorList([ys])
    filter_reg = TensorList([reg])
    sample_weights = TensorList([sw])
    variable = TensorList([var.copy()])

    problem = ConvProblem(training_samples, y, filter_reg, sample_weights,
                          response_activation=layers.relu)

    optimizer = optim_static.GradientDescentL2(problem, variable, debug=True,
                                               step_length=1e-2, momentum=0)
    optimizer.run(10)

    training_samples = TensorListT([n2t(np.stack(ts))])
    y = TensorListT([n2t(ys)])
    filter_reg = TensorListT([n2t(reg)])
    sample_weights = TensorListT([n2t(np.stack(sw))])

    variable = TensorListT([n2t(var.copy())])
    variable.requires_grad_(True)

    problem = ConvProblemT(training_samples, y, filter_reg, sample_weights,
                           response_activation=torch.relu)

    optimizer = optim_t.GradientDescentL2(problem, variable, debug=True,
                                          step_length=1e-2, momentum=0)
    optimizer.run(10)

def test_ConvProblem_conjugate_gradient_descent():
    rng = np.random.RandomState(0)
    ts = [rng.uniform(-1, 1, [32, 3, 3]).astype('float32') for _ in range(10)]
    ys = [rng.uniform(-1, 1, [1, 3, 3]).astype('float32') for _ in range(10)]
    sw = [rng.uniform(0, 1, [1]).astype('float32') for _ in range(10)]
    reg = rng.uniform(0, 1)
    var = rng.uniform(-1, 1, [1, 32, 3, 3]).astype('float32')

    # Paddle run
    training_samples = TensorList([ts])
    y = TensorList([ys])
    filter_reg = TensorList([reg])
    sample_weights = TensorList([sw])
    variable = TensorList([var.copy()])

    problem = ConvProblem(training_samples, y, filter_reg, sample_weights,
                          response_activation=layers.relu)

    optimizer = optim_static.ConjugateGradient(problem, variable, analyze=True)
    optimizer.run(10)

    # Torch Run
    training_samples = TensorListT([n2t(np.stack(ts))])
    y = TensorListT([n2t(ys)])
    filter_reg = TensorListT([n2t(reg)])
    sample_weights = TensorListT([n2t(np.stack(sw))])

    variable = TensorListT([n2t(var.copy())])
    variable.requires_grad_(True)

    problem = ConvProblemT(training_samples, y, filter_reg, sample_weights,
                           response_activation=torch.relu)

    optimizer = optim_t.ConjugateGradient(problem, variable, debug=True, analyze=True)
    optimizer.run(10)


def test_FactorizedConvProblem_conjugate_gradient_descent():
    rng = np.random.RandomState(0)
    ts = [rng.uniform(-1, 1, [32, 3, 3]).astype('float32') for _ in range(10)]
    ys = [rng.uniform(-1, 1, [1, 3, 3]).astype('float32') for _ in range(10)]
    sw = [rng.uniform(0, 1, [1]).astype('float32') for _ in range(10)]
    reg = rng.uniform(0, 1)
    reg2 = rng.uniform(0, 1)
    var1 = rng.uniform(-1, 1, [1, 16, 3, 3]).astype('float32')
    var2 = rng.uniform(-1, 1, [16, 32, 1, 1]).astype('float32')

    # Paddle run
    training_samples = TensorList([ts])
    y = TensorList([ys])
    filter_reg = TensorList([reg])
    project_reg = TensorList([reg2])
    sample_weights = TensorList([sw])
    variable = TensorList([var1.copy(), var2.copy()])

    act_param = 0.05
    # mlu = lambda x: elu(leaky_relu(x, 1 / act_param), act_param)
    # mlu = lambda x: leaky_relu(x, 1 / act_param)
    mlu = lambda x: layers.relu(x)
    problem = FactorizedConvProblem(training_samples, y,
                                    filter_reg=filter_reg,
                                    projection_reg=project_reg,
                                    params=None,
                                    sample_weights=sample_weights,
                                    projection_activation=lambda x: x,
                                    response_activation=mlu)

    optimizer = optim_static.ConjugateGradient(problem, variable, analyze=True)
    optimizer.run(10)

    # Torch Run
    training_samples = TensorListT([n2t(np.stack(ts))])
    y = TensorListT([n2t(ys)])
    filter_reg = TensorListT([n2t(reg, dtype='float32')])
    project_reg = TensorListT([n2t(reg2, dtype='float32')])
    sample_weights = TensorListT([n2t(np.stack(sw))])

    variable = TensorListT([n2t(var1.copy()), n2t(var2.copy())])
    variable.requires_grad_(True)

    # mlu = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
    mlu = lambda x: F.relu(x)
    problem = FactorizedConvProblemT(training_samples, y,
                                     filter_reg=filter_reg,
                                     projection_reg=project_reg,
                                     params=None,
                                     sample_weights=sample_weights,
                                     projection_activation=lambda x: x,
                                     response_activation=mlu)

    optimizer = optim_t.ConjugateGradient(problem, variable, debug=True, analyze=True)
    optimizer.run(10)


def test_FactorizedConvProblem_GaussNewton():
    rng = np.random.RandomState(0)
    for kernel_size in [3, 4, 5]:
        print('kernel size: {}'.format(kernel_size))
        ts = [rng.uniform(-1, 1, [32, 3, 3]).astype('float32') for _ in range(10)]
        ys = [rng.uniform(-1, 1, [1, 3, 3]).astype('float32') for _ in range(10)]
        sw = [rng.uniform(0, 1, [1]).astype('float32') for _ in range(10)]
        reg = rng.uniform(0, 1)
        reg2 = rng.uniform(0, 1)
        var1 = rng.uniform(-1, 1, [1, 16, kernel_size, kernel_size]).astype('float32')
        var2 = rng.uniform(-1, 1, [16, 32, 1, 1]).astype('float32')

        # Paddle run
        training_samples = TensorList([ts])
        y = TensorList([ys])
        filter_reg = TensorList([reg])
        project_reg = TensorList([reg2])
        sample_weights = TensorList([sw])
        variable = TensorList([var1.copy(), var2.copy()])

        act_param = 0.05
        # mlu = lambda x: layers.elu(leaky_relu(x, 1 / act_param), act_param)
        # mlu = lambda x: leaky_relu(x, 1 / act_param)
        # mlu = lambda x: layers.relu(x)
        mlu = lambda x: layers.elu(x)
        problem = FactorizedConvProblem(training_samples, y,
                                        filter_reg=filter_reg,
                                        projection_reg=project_reg,
                                        params=None,
                                        sample_weights=sample_weights,
                                        projection_activation=lambda x: x,
                                        response_activation=mlu)

        optimizer = optim_static.GaussNewtonCG(problem, variable, analyze=True)
        optimizer.run(10, 2)
        print('Paddle filter: {}'.format(variable.mean()))

        # Torch Run
        training_samples = TensorListT([n2t(np.stack(ts))])
        y = TensorListT([n2t(ys)])
        filter_reg = TensorListT([n2t(reg, dtype='float32')])
        project_reg = TensorListT([n2t(reg2, dtype='float32')])
        sample_weights = TensorListT([n2t(np.stack(sw))])

        variable = TensorListT([n2t(var1.copy()), n2t(var2.copy())])
        variable.requires_grad_(True)

        # mlu = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        # mlu = lambda x: F.relu(x)
        mlu = lambda x: F.elu(x)
        problem = FactorizedConvProblemT(training_samples, y,
                                         filter_reg=filter_reg,
                                         projection_reg=project_reg,
                                         params=None,
                                         sample_weights=sample_weights,
                                         projection_activation=lambda x: x,
                                         response_activation=mlu)

        optimizer = optim_t.GaussNewtonCG(problem, variable, debug=True, analyze=True)
        optimizer.run(10, 2)
        print('Torch filter: {}'.format(variable.mean()))

def test_FactorizedConvProblem_GaussNewton_real():
    def load(fname):
        import pickle
        with open(fname, 'rb') as f:
            return pickle.load(f)

    # Paddle run
    training_samples = load('x.pickle')
    y = load('y.pickle')
    filter_reg = TensorList([1e-1])
    project_reg = TensorList([1e-4])
    sample_weights = load('sample_weights.pickle')
    variable = load('joint_var.pickle')

    act_param = 0.05
    # mlu = lambda x: elu(leaky_relu(x, 1 / act_param), act_param)
    # mlu = lambda x: leaky_relu(x, 1 / act_param)
    mlu = lambda x: layers.relu(x)
    problem = FactorizedConvProblem(training_samples, y,
                                    filter_reg=filter_reg,
                                    projection_reg=project_reg,
                                    params=None,
                                    sample_weights=sample_weights,
                                    projection_activation=lambda x: x,
                                    response_activation=mlu)

    optimizer = optim_static.GaussNewtonCG(problem, variable, analyze=True)
    optimizer.run(10, 6)
    print('Paddle filter: {}'.format(variable.mean()))


def test_gradient():
    rng = np.random.RandomState(0)
    ts = [rng.uniform(-1, 1, [32, 3, 3]).astype('float32') for _ in range(10)]
    ys = [rng.uniform(-1, 1, [1, 3, 3]).astype('float32') for _ in range(10)]
    sw = [rng.uniform(0, 1, [1]).astype('float32') for _ in range(10)]
    reg = rng.uniform(0, 1)
    var = rng.uniform(-1, 1, [1, 32, 3, 3]).astype('float32')

    training_samples = TensorList([ts])
    y = TensorList([ys])
    filter_reg = TensorList([reg])
    sample_weights = TensorList([sw])
    variable = TensorList([var])

    problem = ConvProblem(training_samples, y, filter_reg, sample_weights,
                          response_activation=layers.relu)
    optimizer = optim_static.ConjugateGradient(problem, variable)
    b_paddle = - optimizer.get_dfdxt_g()
    self = optimizer
    self.b = b_paddle

    x = None
    num_iter = 1

    if self.direction_forget_factor == 0:
        self.reset_state()
    elif self.p is not None:
        self.rho /= self.direction_forget_factor

    if x is None:
        r = self.b.copy()
    else:
        r = self.b - self.A(x)

    # Norms of residuals etc for debugging
    resvec = None

    # Loop over iterations
    for ii in range(num_iter):
        # Preconditioners
        y = self.M1(r)
        z = self.M2(y)

        rho1 = self.rho
        self.rho = self.ip(r, z)

        if self.check_zero(self.rho):
            if self.debug:
                print('Stopped CG since rho = 0')
                if resvec is not None:
                    resvec = resvec[:ii + 1]
            return x, resvec

        if self.p is None:
            self.p = z.copy()
        else:
            if self.fletcher_reeves:
                beta = self.rho / rho1
            else:
                rho2 = self.ip(self.r_prev, z)
                beta = (self.rho - rho2) / rho1

            beta = beta.apply(lambda a: np.clip(a, 0, 1e10))
            self.p = z + self.p * beta

        q = self.A(self.p)
        # x = self.p
        # feed_dict = self.problem.get_feed_dict()
        # # add variable feed
        # for idx, v in enumerate(self.x):
        #     feed_dict['x_{}'.format(idx)] = v
        # # add p feed
        # for idx, v in enumerate(x):
        #     feed_dict['p_{}'.format(idx)] = v
        #
        # dfdx_x_paddle = self.exe.run(self.compiled_prog,
        #                              feed=feed_dict,
        #                              fetch_list=[v.name for v in self.dfdx_x])
        # q = self.exe.run(self.compiled_prog,
        #                  feed=feed_dict,
        #                  fetch_list=[v.name for v in self.dfdx_dfdx])

        q_paddle = q

    ### Pytorch
    training_samples = TensorListT([n2t(np.stack(ts))])
    y = TensorListT([n2t(ys)])
    filter_reg = TensorListT([n2t(reg)])
    sample_weights = TensorListT([n2t(np.stack(sw))])

    variable = TensorListT([n2t(var)])
    variable.requires_grad_(True)

    problem = ConvProblemT(training_samples, y, filter_reg, sample_weights,
                           response_activation=torch.relu)

    optimizer = optim_t.ConjugateGradient(problem, variable, debug=True)

    self = optimizer
    x = None
    self.x.requires_grad_(True)
    # Evaluate function at current estimate
    self.f0 = self.problem(self.x)
    # Create copy with graph detached
    self.g = self.f0.detach()
    self.g.requires_grad_(True)

    # Get df/dx^t @ f0
    self.dfdxt_g = TensorListT(torch.autograd.grad(self.f0, self.x, self.g, create_graph=True))
    # Get the right hand side
    b_torch = - self.dfdxt_g.detach().numpy()

    self.b = - self.dfdxt_g.detach()

    # Apply forgetting factor
    if self.direction_forget_factor == 0:
        self.reset_state()
    elif self.p is not None:
        self.rho /= self.direction_forget_factor

    if x is None:
        r = self.b.clone()
    else:
        r = self.b - self.A(x)

    # Norms of residuals etc for debugging
    resvec = None
    if self.debug:
        normr = self.residual_norm(r)
        resvec = torch.zeros(num_iter + 1)
        resvec[0] = normr

    # Loop over iterations
    for ii in range(num_iter):
        # Preconditioners
        y = self.M1(r)
        z = self.M2(y)

        rho1 = self.rho
        self.rho = self.ip(r, z)

        if self.check_zero(self.rho):
            if self.debug:
                print('Stopped CG since rho = 0')
                if resvec is not None:
                    resvec = resvec[:ii + 1]
            return x, resvec

        if self.p is None:
            self.p = z.clone()
        else:
            if self.fletcher_reeves:
                beta = self.rho / rho1
            else:
                rho2 = self.ip(self.r_prev, z)
                beta = (self.rho - rho2) / rho1

            beta = beta.clamp(0)
            self.p = z + self.p * beta

        # q = self.A(self.p)
        x = self.p
        dfdx_x_torch = torch.autograd.grad(self.dfdxt_g, self.g, x, retain_graph=True)
        q = TensorListT(torch.autograd.grad(self.f0, self.x, dfdx_x_torch, retain_graph=True))
        q_torch = q.detach().cpu().numpy()

    ### Assert equal
    for p, t in zip(b_paddle, b_torch):
        np.testing.assert_allclose(p, t, rtol=1e-04, atol=1e-05)
    for p, t in zip(q_paddle, q_torch):
        np.testing.assert_allclose(p, t, rtol=1e-04, atol=1e-05)


def test_construct_graph():
    rng = np.random.RandomState(0)
    ts = [rng.uniform(-1, 1, [32, 3, 3]).astype('float32') for _ in range(10)]
    ys = [rng.uniform(-1, 1, [1, 3, 3]).astype('float32') for _ in range(10)]
    sw = [rng.uniform(0, 1, [1]).astype('float32') for _ in range(10)]
    reg = rng.uniform(0, 1)
    var = rng.uniform(-1, 1, [1, 32, 3, 3]).astype('float32')

    training_samples = [np.stack(ts)]
    y = [np.stack(ys)]
    filter_reg = [reg]
    sample_weights = [np.stack(sw)]
    variable = [var]

    x = variable

    train_program = fluid.Program()
    start_program = fluid.Program()
    with fluid.program_guard(train_program, start_program):
        x_ph = TensorList(
            [fluid.layers.data('x_{}'.format(idx), v.shape,
                               append_batch_size=False,
                               stop_gradient=False)
             for idx, v in enumerate(x)])
        p_ph = TensorList(
            [fluid.layers.data('p_{}'.format(idx), v.shape,
                               append_batch_size=False,
                               stop_gradient=False)
             for idx, v in enumerate(x)])

        # problem forward
        f0 = problem(x_ph)

        g = f0.apply(static_clone)
        # self.g = self.f0

        # Get df/dx^t @ f0
        self.dfdxt_g = TensorList(fluid.gradients(self.f0, self.x_ph, self.g))

        # For computing A
        tmp = [a * b for a, b in zip(self.dfdxt_g, self.p_ph)]
        self.dfdx_x = TensorList(fluid.gradients(tmp, self.g))
        # self.dfdx_x = TensorList(fluid.gradients(self.dfdxt_g, self.g, self.p_ph))

        self.dfdx_dfdx = TensorList(fluid.gradients(self.f0, self.x_ph, self.dfdx_x))


if __name__ == '__main__':
    # test_ConvProblem_conjugate_gradient_descent()
    # test_FactorizedConvProblem_conjugate_gradient_descent()
    test_FactorizedConvProblem_GaussNewton()
    # test_FactorizedConvProblem_GaussNewton_real()
    # test_ConvProblem_gradient_descent()
    # test_gradient()
