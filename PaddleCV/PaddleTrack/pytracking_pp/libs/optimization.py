import numpy as np
from paddle.fluid import layers
from paddle import fluid
from pytracking_pp.libs.tensorlist import TensorList
from pytracking_pp.utils.plotting import plot_graph
from pytracking_pp.libs.paddle_utils import n2p, clone, static_clone


class L2Problem:
    """Base class for representing an L2 optimization problem."""

    def __call__(self, x: TensorList) -> TensorList:
        """Shall compute the residuals of the problem."""
        raise NotImplementedError

    def ip_input(self, a, b):
        """Inner product of the input space."""
        return sum(a.view(-1) @ b.view(-1))

    def ip_output(self, a, b):
        """Inner product of the output space."""
        return sum(a.view(-1) @ b.view(-1))

    def M1(self, x):
        """M1 preconditioner."""
        return x

    def M2(self, x):
        """M2 preconditioner."""
        return x

    def get_feed_dict(self):
        raise NotImplementedError


class MinimizationProblem:
    """General minimization problem."""

    def __call__(self, x: TensorList) -> TensorList:
        """Shall compute the loss."""
        raise NotImplementedError

    def ip_input(self, a, b):
        """Inner product of the input space."""
        return sum(a.view(-1) @ b.view(-1))

    def M1(self, x):
        return x

    def M2(self, x):
        return x

    def get_feed_dict(self):
        raise NotImplementedError


class ConjugateGradientBase:
    """Conjugate Gradient optimizer base class. Implements the CG loop."""

    def __init__(self, fletcher_reeves=True, standard_alpha=True, direction_forget_factor=0, debug=False):
        self.fletcher_reeves = fletcher_reeves
        self.standard_alpha = standard_alpha
        self.direction_forget_factor = direction_forget_factor
        self.debug = debug

        # State
        self.p = None
        self.rho = np.ones((1,), 'float32')
        self.r_prev = None

        # Right hand side
        self.b = None

    def reset_state(self):
        self.p = None
        self.rho = np.ones((1,), 'float32')
        self.r_prev = None

    def run_CG(self, num_iter, x=None, eps=0.0):
        """Main conjugate gradient method.

        args:
            num_iter: Number of iterations.
            x: Initial guess. Assumed zero if None.
            eps: Stop if the residual norm gets smaller than this.
        """

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

                beta = beta.apply(lambda a: np.clip(a, 0, 1e10))
                self.p = z + self.p * beta

            q = self.A(self.p)
            pq = self.ip(self.p, q)

            if self.standard_alpha:
                alpha = self.rho / pq
            else:
                alpha = self.ip(self.p, r) / pq

            # Save old r for PR formula
            if not self.fletcher_reeves:
                self.r_prev = r.clone()

            # Form new iterate
            if x is None:
                x = self.p * alpha
            else:
                x += self.p * alpha

            if ii < num_iter - 1 or self.debug:
                r -= q * alpha

            if eps > 0.0 or self.debug:
                normr = self.residual_norm(r)

            # if self.debug:
            if True:
                self.evaluate_CG_iteration(x)
                # resvec[ii + 1] = normr

            if eps > 0 and normr <= eps:
                if self.debug:
                    print('Stopped CG since norm smaller than eps')
                break

        if resvec is not None:
            resvec = resvec[:ii + 2]

        return x, resvec

    def A(self, x):
        # Implements the left hand operation
        raise NotImplementedError

    def ip(self, a, b):
        # Implements the inner product
        return a.view(-1) @ b.view(-1)

    def residual_norm(self, r):
        res = self.ip(r, r).sum()
        if isinstance(res, (TensorList, list, tuple)):
            res = sum(res)
        return np.sqrt(res)

    def check_zero(self, s, eps=0.0):
        ss = s.abs() <= eps
        if isinstance(ss, (TensorList, list, tuple)):
            ss = sum(ss)
        return ss > 0

    def M1(self, x):
        # M1 preconditioner
        return x

    def M2(self, x):
        # M2 preconditioner
        return x

    def evaluate_CG_iteration(self, x):
        pass


class ConjugateGradient(ConjugateGradientBase):
    """Conjugate Gradient optimizer, performing single linearization of the residuals in the start."""

    def __init__(self, problem: L2Problem, variable: TensorList, cg_eps=0.0, fletcher_reeves=True,
                 standard_alpha=True, direction_forget_factor=0, debug=False, analyze=False, plotting=False,
                 fig_num=(10, 11)):
        super().__init__(fletcher_reeves, standard_alpha, direction_forget_factor, debug or plotting)

        self.problem = problem
        self.x = variable

        self.plotting = plotting
        self.fig_num = fig_num

        self.cg_eps = cg_eps
        self.f0 = None
        self.g = None
        self.dfdxt_g = None

        self.residuals = np.zeros(0)
        self.losses = np.zeros(0)
        self._construct_graph()
        self.analyze_convergence = analyze

    def clear_temp(self):
        pass

    def _construct_graph(self):
        train_program = fluid.Program()
        start_program = fluid.Program()
        with fluid.program_guard(train_program, start_program):
            scope = 'first/'
            self.x_ph = TensorList(
                [fluid.layers.data('{}x_{}'.format(scope, idx), v.shape,
                                   append_batch_size=False,
                                   stop_gradient=False)
                 for idx, v in enumerate(self.x)])
            self.p_ph = TensorList(
                [fluid.layers.data('{}p_{}'.format(scope, idx), v.shape,
                                   append_batch_size=False,
                                   stop_gradient=False)
                 for idx, v in enumerate(self.x)])

            # problem forward
            self.f0 = self.problem(self.x_ph, scope)

            self.g = self.f0.apply(static_clone)
            # self.g = self.f0

            # Get df/dx^t @ f0
            self.dfdxt_g = TensorList(fluid.gradients(self.f0, self.x_ph, self.g))

            # For computing A
            tmp = [a * b for a, b in zip(self.dfdxt_g, self.p_ph)]
            self.dfdx_x = TensorList(fluid.gradients(tmp, self.g))
            # self.dfdx_x = TensorList(fluid.gradients(self.dfdxt_g, self.g, self.p_ph))

        train_program2 = fluid.Program()
        start_program2 = fluid.Program()
        with fluid.program_guard(train_program2, start_program2):
            scope = 'second/'
            self.x_ph_2 = TensorList(
                [fluid.layers.data('{}x_{}'.format(scope, idx), v.shape,
                                   append_batch_size=False,
                                   stop_gradient=False)
                 for idx, v in enumerate(self.x)])
            self.dfdx_x_ph = TensorList(
                [fluid.layers.data('{}dfdx_x_{}'.format(scope, idx), v.shape,
                                   append_batch_size=False,
                                   stop_gradient=False)
                 for idx, v in enumerate(self.g)])

            self.f0_2 = self.problem(self.x_ph_2, scope)
            self.dfdx_dfdx = TensorList(fluid.gradients(self.f0_2 * self.dfdx_x_ph, self.x_ph_2))

        place = fluid.CUDAPlace(0)
        self.exe = fluid.Executor(place)
        self.exe.run(program=fluid.default_startup_program())
        self.compiled_prog = fluid.compiler.CompiledProgram(train_program)

        place2 = fluid.CUDAPlace(0)
        self.exe2 = fluid.Executor(place2)
        self.exe2.run(program=fluid.default_startup_program())
        self.compiled_prog2 = fluid.compiler.CompiledProgram(train_program2)

    def get_dfdxt_g(self):
        scope = 'first/'
        feed_dict = self.problem.get_feed_dict(scope)
        # add variable feed
        for idx, v in enumerate(self.x):
            feed_dict['{}x_{}'.format(scope, idx)] = v
        for idx, v in enumerate(self.x):
            feed_dict['{}p_{}'.format(scope, idx)] = v
        res = self.exe.run(self.compiled_prog, feed=feed_dict, fetch_list=[v.name for v in self.dfdxt_g])
        return TensorList(res)

    def run(self, num_cg_iter):
        """Run the oprimizer with the provided number of iterations."""

        if num_cg_iter == 0:
            return

        # Get the right hand side
        self.b = - self.get_dfdxt_g()

        self.evaluate_CG_iteration(0)

        # Run CG
        delta_x, res = self.run_CG(num_cg_iter, eps=self.cg_eps)

        self.x += delta_x

        # reset problem training samples
        self.problem.training_samples_stack = None


    def A(self, x):
        # First pass
        scope = 'first/'
        feed_dict = self.problem.get_feed_dict(scope)
        # add variable feed
        for idx, v in enumerate(self.x):
            feed_dict['{}x_{}'.format(scope, idx)] = v
        # add p feed
        for idx, v in enumerate(x):
            feed_dict['{}p_{}'.format(scope, idx)] = v

        dfdx_x = TensorList(self.exe.run(self.compiled_prog,
                                         feed=feed_dict,
                                         fetch_list=[v.name for v in self.dfdx_x]))

        # Second pass
        scope = 'second/'
        feed_dict = self.problem.get_feed_dict(scope)
        # add variable feed
        for idx, v in enumerate(self.x):
            feed_dict['{}x_{}'.format(scope, idx)] = v
        # add p feed
        for idx, v in enumerate(dfdx_x):
            feed_dict['{}dfdx_x_{}'.format(scope, idx)] = v

        res = TensorList(self.exe2.run(self.compiled_prog2,
                                       feed=feed_dict,
                                       fetch_list=[v.name for v in self.dfdx_dfdx]))

        return res

    def ip(self, a, b):
        return self.problem.ip_input(a, b)

    def M1(self, x):
        return self.problem.M1(x)

    def M2(self, x):
        return self.problem.M2(x)

    def evaluate_CG_iteration(self, delta_x):
        if self.analyze_convergence:
            scope = 'first/'
            x = self.x + delta_x
            feed_dict = self.problem.get_feed_dict(scope)
            for idx, v in enumerate(x):
                feed_dict['{}x_{}'.format(scope, idx)] = v
            for idx, v in enumerate(x):
                feed_dict['{}p_{}'.format(scope, idx)] = v
            res = self.exe.run(self.compiled_prog,
                               feed=feed_dict,
                               fetch_list=[v.name for v in self.f0])
            res = TensorList(res)
            loss = self.problem.ip_output(res, res)
            print('Paddle Loss: {}'.format(loss))


class GaussNewtonCG(ConjugateGradientBase):
    """Gauss-Newton with Conjugate Gradient optimizer."""

    def __init__(self, problem: L2Problem, variable: TensorList, cg_eps=0.0, fletcher_reeves=True,
                 standard_alpha=True, direction_forget_factor=0, debug=False, analyze=False, plotting=False,
                 fig_num=(10, 11, 12)):
        super().__init__(fletcher_reeves, standard_alpha, direction_forget_factor, debug or analyze or plotting)

        self.problem = problem
        self.x = variable

        self.analyze_convergence = analyze
        self.plotting = plotting
        self.fig_num = fig_num

        self.cg_eps = cg_eps
        self.f0 = None
        self.g = None
        self.dfdxt_g = None

        self.residuals = np.zeros(0)
        self.losses = np.zeros(0)
        self.gradient_mags = np.zeros(0)
        self._construct_graph()

    def clear_temp(self):
        self.f0 = None
        self.g = None
        self.dfdxt_g = None

    def run_GN(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def _construct_graph(self):
        train_program = fluid.Program()
        start_program = fluid.Program()
        with fluid.program_guard(train_program, start_program):
            scope = 'first/'
            self.x_ph = TensorList(
                [fluid.layers.data('{}x_{}'.format(scope, idx), v.shape,
                                   append_batch_size=False,
                                   stop_gradient=False)
                 for idx, v in enumerate(self.x)])
            self.p_ph = TensorList(
                [fluid.layers.data('{}p_{}'.format(scope, idx), v.shape,
                                   append_batch_size=False,
                                   stop_gradient=False)
                 for idx, v in enumerate(self.x)])

            # problem forward
            self.f0 = self.problem(self.x_ph, scope)

            self.g = self.f0.apply(static_clone)

            # Get df/dx^t @ f0
            self.dfdxt_g = TensorList(fluid.gradients(self.f0, self.x_ph, self.g))

            # For computing A
            tmp = [a * b for a, b in zip(self.dfdxt_g, self.p_ph)]
            self.dfdx_x = TensorList(fluid.gradients(tmp, self.g))
            # self.dfdx_x = TensorList(fluid.gradients(self.dfdxt_g, self.g, self.p_ph))

        train_program2 = fluid.Program()
        start_program2 = fluid.Program()
        with fluid.program_guard(train_program2, start_program2):
            scope = 'second/'
            self.x_ph_2 = TensorList(
                [fluid.layers.data('{}x_{}'.format(scope, idx), v.shape,
                                   append_batch_size=False,
                                   stop_gradient=False)
                 for idx, v in enumerate(self.x)])
            self.dfdx_x_ph = TensorList(
                [fluid.layers.data('{}dfdx_x_{}'.format(scope, idx), v.shape,
                                   append_batch_size=False,
                                   stop_gradient=False)
                 for idx, v in enumerate(self.g)])

            self.f0_2 = self.problem(self.x_ph_2, scope)
            self.dfdx_dfdx = TensorList(fluid.gradients(self.f0_2 * self.dfdx_x_ph, self.x_ph_2))

        place = fluid.CUDAPlace(0)
        self.exe = fluid.Executor(place)
        self.exe.run(program=fluid.default_startup_program())
        self.compiled_prog = fluid.compiler.CompiledProgram(train_program)

        place2 = fluid.CUDAPlace(0)
        self.exe2 = fluid.Executor(place2)
        self.exe2.run(program=fluid.default_startup_program())
        self.compiled_prog2 = fluid.compiler.CompiledProgram(train_program2)

    def get_dfdxt_g(self):
        scope = 'first/'
        feed_dict = self.problem.get_feed_dict(scope)
        # add variable feed
        for idx, v in enumerate(self.x):
            feed_dict['{}x_{}'.format(scope, idx)] = v
        for idx, v in enumerate(self.x):
            feed_dict['{}p_{}'.format(scope, idx)] = v
        res = self.exe.run(self.compiled_prog, feed=feed_dict, fetch_list=[v.name for v in self.dfdxt_g])
        return TensorList(res)

    def run(self, num_cg_iter, num_gn_iter=None):
        """Run the optimizer.
        args:
            num_cg_iter: Number of CG iterations per GN iter. If list, then each entry specifies number of CG iterations
                         and number of GN iterations is given by the length of the list.
            num_gn_iter: Number of GN iterations. Shall only be given if num_cg_iter is an integer.
        """

        if isinstance(num_cg_iter, int):
            if num_gn_iter is None:
                raise ValueError('Must specify number of GN iter if CG iter is constant')
            num_cg_iter = [num_cg_iter] * num_gn_iter

        num_gn_iter = len(num_cg_iter)
        if num_gn_iter == 0:
            return

        if self.analyze_convergence:
            self.evaluate_CG_iteration(0)

        # Outer loop for running the GN iterations.
        for cg_iter in num_cg_iter:
            self.run_GN_iter(cg_iter)

        # reset problem training samples
        self.problem.training_samples_stack = None

        # if self.debug:
        #     if not self.analyze_convergence:
        #         self.f0 = self.problem(self.x)
        #         loss = self.problem.ip_output(self.f0, self.f0)
        #         self.losses = torch.cat((self.losses, loss.detach().cpu().view(-1)))
        #
        #     if self.plotting:
        #         plot_graph(self.losses, self.fig_num[0], title='Loss')
        #         plot_graph(self.residuals, self.fig_num[1], title='CG residuals')
        #         if self.analyze_convergence:
        #             plot_graph(self.gradient_mags, self.fig_num[2], 'Gradient magnitude')

        return self.losses, self.residuals

    def run_GN_iter(self, num_cg_iter):
        """Runs a single GN iteration."""

        # if self.debug and not self.analyze_convergence:
        #     loss = self.problem.ip_output(self.g, self.g)
        #     self.losses = torch.cat((self.losses, loss.detach().cpu().view(-1)))

        self.b = - self.get_dfdxt_g()

        # Run CG
        if num_cg_iter > 0:
            delta_x, res = self.run_CG(num_cg_iter, eps=self.cg_eps)
            self.x += delta_x

        # if self.debug:
        #     self.residuals = torch.cat((self.residuals, res))

    def A(self, x):
        # First pass
        scope = 'first/'
        feed_dict = self.problem.get_feed_dict(scope)
        # add variable feed
        for idx, v in enumerate(self.x):
            feed_dict['{}x_{}'.format(scope, idx)] = v
        # add p feed
        for idx, v in enumerate(x):
            feed_dict['{}p_{}'.format(scope, idx)] = v

        dfdx_x = TensorList(self.exe.run(self.compiled_prog,
                                         feed=feed_dict,
                                         fetch_list=[v.name for v in self.dfdx_x]))

        # Second pass
        scope = 'second/'
        feed_dict = self.problem.get_feed_dict(scope)
        # add variable feed
        for idx, v in enumerate(self.x):
            feed_dict['{}x_{}'.format(scope, idx)] = v
        # add p feed
        for idx, v in enumerate(dfdx_x):
            feed_dict['{}dfdx_x_{}'.format(scope, idx)] = v

        res = TensorList(self.exe2.run(self.compiled_prog2,
                                       feed=feed_dict,
                                       fetch_list=[v.name for v in self.dfdx_dfdx]))

        return res

    def ip(self, a, b):
        return self.problem.ip_input(a, b)

    def M1(self, x):
        return self.problem.M1(x)

    def M2(self, x):
        return self.problem.M2(x)

    def evaluate_CG_iteration(self, delta_x):
        if self.analyze_convergence:
            scope = 'first/'
            x = self.x + delta_x
            feed_dict = self.problem.get_feed_dict(scope)
            for idx, v in enumerate(x):
                feed_dict['{}x_{}'.format(scope, idx)] = v
            for idx, v in enumerate(x):
                feed_dict['{}p_{}'.format(scope, idx)] = v
            res = self.exe.run(self.compiled_prog,
                               feed=feed_dict,
                               fetch_list=[v.name for v in self.f0])
            res = TensorList(res)
            loss = self.problem.ip_output(res, res)
            print('Paddle Loss: {}'.format(loss))


class NewtonCG(ConjugateGradientBase):
    """Newton with Conjugate Gradient. Handels general minimization problems."""

    def __init__(self, problem: MinimizationProblem, variable: TensorList, init_hessian_reg=0.0, hessian_reg_factor=1.0,
                 cg_eps=0.0, fletcher_reeves=True, standard_alpha=True, direction_forget_factor=0,
                 debug=False, analyze=False, plotting=False, fig_num=(10, 11, 12)):
        super().__init__(fletcher_reeves, standard_alpha, direction_forget_factor, debug or analyze or plotting)

        self.problem = problem
        self.x = variable

        self.analyze_convergence = analyze
        self.plotting = plotting
        self.fig_num = fig_num

        self.hessian_reg = init_hessian_reg
        self.hessian_reg_factor = hessian_reg_factor
        self.cg_eps = cg_eps
        self.f0 = None
        self.g = None

        self.residuals = np.zeros(0)
        self.losses = np.zeros(0)
        self.gradient_mags = np.zeros(0)

    def clear_temp(self):
        self.f0 = None
        self.g = None

    def run(self, num_cg_iter, num_newton_iter=None):

        if isinstance(num_cg_iter, int):
            if num_cg_iter == 0:
                return
            if num_newton_iter is None:
                num_newton_iter = 1
            num_cg_iter = [num_cg_iter] * num_newton_iter

        num_newton_iter = len(num_cg_iter)
        if num_newton_iter == 0:
            return

        if self.analyze_convergence:
            self.evaluate_CG_iteration(0)

        for cg_iter in num_cg_iter:
            self.run_newton_iter(cg_iter)
            self.hessian_reg *= self.hessian_reg_factor

        self.x.detach_()
        self.clear_temp()

        return self.losses, self.residuals

    def run_newton_iter(self, num_cg_iter):

        self.x.requires_grad_(True)

        # Evaluate function at current estimate
        self.f0 = self.problem(self.x)

        if self.debug and not self.analyze_convergence:
            self.losses = torch.cat((self.losses, self.f0.detach().cpu().view(-1)))

        # Gradient of loss
        self.g = TensorList(torch.autograd.grad(self.f0, self.x, create_graph=True))

        # Get the right hand side
        self.b = - self.g.detach()

        # Run CG
        delta_x, res = self.run_CG(num_cg_iter, eps=self.cg_eps)

        self.x.detach_()
        self.x += delta_x

        if self.debug:
            self.residuals = torch.cat((self.residuals, res))

    def A(self, x):
        return TensorList(torch.autograd.grad(self.g, self.x, x, retain_graph=True)) + self.hessian_reg * x

    def ip(self, a, b):
        # Implements the inner product
        return self.problem.ip_input(a, b)

    def M1(self, x):
        return self.problem.M1(x)

    def M2(self, x):
        return self.problem.M2(x)

    def evaluate_CG_iteration(self, delta_x):
        if self.analyze_convergence:
            x = (self.x + delta_x).detach()
            x.requires_grad_(True)

            # compute loss and gradient
            loss = self.problem(x)
            grad = TensorList(torch.autograd.grad(loss, x))

            # store in the vectors
            self.losses = torch.cat((self.losses, loss.detach().cpu().view(-1)))
            self.gradient_mags = torch.cat(
                (self.gradient_mags, sum(grad.view(-1) @ grad.view(-1)).cpu().sqrt().detach().view(-1)))


class GradientDescentL2:
    """Gradient descent with momentum for L2 problems."""

    def __init__(self, problem: L2Problem, variable: TensorList,
                 step_length: float, momentum: float = 0.0, debug=False,
                 plotting=False, fig_num=(10, 11)):

        self.problem = problem
        self.x = variable  # Numpy arrays

        self.step_legnth = step_length
        self.momentum = momentum

        self.debug = debug or plotting
        self.plotting = plotting
        self.fig_num = fig_num

        self.losses = np.zeros(0)
        self.gradient_mags = np.zeros(0)
        self.residuals = None

        self.clear_temp()
        self._construct_graph()

    def clear_temp(self):
        self.f0 = None
        self.dir = None

    def _construct_graph(self):
        train_program = fluid.Program()
        start_program = fluid.Program()
        with fluid.program_guard(train_program, start_program):
            self.x_ph = TensorList(
                [fluid.layers.data('x_{}'.format(idx), v.shape, append_batch_size=False, stop_gradient=False)
                 for idx, v in enumerate(self.x)])

            # problem forward
            self.f0 = self.problem(self.x_ph)
            self.loss = self.problem.ip_output(self.f0, self.f0)
            # problem backward
            self.grad = TensorList(fluid.gradients(self.loss, self.x_ph))

        place = fluid.CUDAPlace(0)
        self.exe = fluid.Executor(place)
        self.exe.run(program=fluid.default_startup_program())
        self.compiled_prog = fluid.compiler.CompiledProgram(train_program)

    def get_feed_dict(self, x_list):
        feed_dict = self.problem.get_feed_dict()
        # add variable feed
        for idx, v in enumerate(x_list):
            feed_dict['x_{}'.format(idx)] = v
        return feed_dict

    def run(self, num_iter, dummy=None):
        if num_iter == 0:
            return

        grad_names = [v.name for v in self.grad]
        for i in range(num_iter):
            res = self.exe.run(self.compiled_prog,
                               feed=self.get_feed_dict(self.x),
                               fetch_list=[self.loss.name] + grad_names)
            if self.debug:
                loss = res[0]
                print('Paddle Loss: {}'.format(loss))

            grad = TensorList(res[1:])

            # update parameters
            if self.dir is None:
                self.dir = grad
            else:
                self.dir = grad + self.momentum * self.dir
            self.x = self.x - self.step_legnth * self.dir

        # reset problem training samples
        self.problem.training_samples_stack = None


class GradientDescent:
    """Gradient descent for general minimization problems."""

    def __init__(self, problem: MinimizationProblem, variable: TensorList, step_length: float, momentum: float = 0.0,
                 debug=False, plotting=False, fig_num=(10, 11)):

        self.problem = problem
        self.x = variable

        self.step_legnth = step_length
        self.momentum = momentum

        self.debug = debug or plotting
        self.plotting = plotting
        self.fig_num = fig_num

        self.losses = layers.zeros((0,), 'float32')
        self.gradient_mags = layers.zeros((0,), 'float32')
        self.residuals = None

        self.clear_temp()

    def clear_temp(self):
        self.dir = None

    def run(self, num_iter, dummy=None):

        if num_iter == 0:
            return

        lossvec = None
        if self.debug:
            lossvec = np.zeros((num_iter + 1,))
            grad_mags = np.zeros((num_iter + 1,))

        for i in range(num_iter):
            self.x.stop_gradient = False

            # Evaluate function at current estimate
            loss = self.problem(self.x)

            # Compute grad
            loss.backward()
            grad = TensorList(self.x.gradient()).apply(n2p)
            self.x.clear_gradient()

            # Update direction
            if self.dir is None:
                self.dir = grad
            else:
                self.dir = grad + self.momentum * self.dir

            self.x = self.x.detach()
            self.x -= self.step_legnth * self.dir

            if self.debug:
                lossvec[i] = loss.numpy()
                grad_mags[i] = self.problem.ip_input(grad, grad).apply(layers.sqrt).numpy()

        # if self.debug:
        #     self.x.requires_grad_(True)
        #     loss = self.problem(self.x)
        #     grad = TensorList(torch.autograd.grad(loss, self.x))
        #     lossvec[-1] = loss.item()
        #     grad_mags[-1] = sum(grad.view(-1) @ grad.view(-1)).cpu().sqrt().item()
        #     self.losses = torch.cat((self.losses, lossvec))
        #     self.gradient_mags = torch.cat((self.gradient_mags, grad_mags))
        #     if self.plotting:
        #         plot_graph(self.losses, self.fig_num[0], title='Loss')
        #         plot_graph(self.gradient_mags, self.fig_num[1], title='Gradient magnitude')

        # self.x.detach_()
        self.problem.training_samples_stack = None

        self.x = self.x.detach()
        self.x.stop_gradient = True
        self.clear_temp()
