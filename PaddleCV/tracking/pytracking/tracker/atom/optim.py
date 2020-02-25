import numpy as np
from paddle.fluid import layers
from paddle import fluid

from pytracking.libs import optimization, TensorList, operation
from pytracking.libs.paddle_utils import PTensor, broadcast_op, n2p, static_identity
import math


def stack_input(e):
    if isinstance(e, list):
        e_exist = []
        for x in e:
            if x is not None:
                e_exist.append(x)
        e = np.stack(e_exist)
    else:
        assert isinstance(e, np.ndarray)
        if len(e.shape) == 1:
            e = np.expand_dims(e, 1)
    return e


class FactorizedConvProblem(optimization.L2Problem):
    def __init__(self,
                 training_samples: TensorList,
                 y: TensorList,
                 filter_reg: TensorList,
                 projection_reg,
                 params,
                 sample_weights: TensorList,
                 projection_activation,
                 response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.params = params
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation
        self.response_activation = response_activation

        self.diag_M = self.filter_reg.concat(projection_reg)

        self.inputs_dict = {}
        # stack tensors
        self.training_samples_stack = None
        self.y_stack = None
        self.sample_weights_stack = None

    def get_inputs(self, scope=''):
        if scope not in self.inputs_dict:
            training_samples_p = TensorList([
                fluid.layers.data(
                    '{}training_samples_{}'.format(scope, idx),
                    shape=[None] + list(v[0].shape),
                    stop_gradient=False,
                    append_batch_size=False)
                for idx, v in enumerate(self.training_samples)
            ])
            y_p = TensorList([
                fluid.layers.data(
                    '{}y_{}'.format(scope, idx),
                    shape=[None] + list(v[0].shape),
                    stop_gradient=False,
                    append_batch_size=False) for idx, v in enumerate(self.y)
            ])
            sample_weights_p = TensorList([
                fluid.layers.data(
                    '{}sample_weights_{}'.format(scope, idx),
                    shape=[None, 1],
                    stop_gradient=False,
                    append_batch_size=False)
                for idx, v in enumerate(self.sample_weights)
            ])
            self.inputs_dict[scope] = (training_samples_p, y_p,
                                       sample_weights_p)

        return self.inputs_dict[scope]

    def get_feed_dict(self, scope=''):
        if self.training_samples_stack is None or self.y_stack is None or self.sample_weights_stack is None:
            self.training_samples_stack = self.training_samples.apply(
                stack_input)
            self.y_stack = self.y.apply(stack_input)
            self.sample_weights_stack = self.sample_weights.apply(stack_input)
        feed_dict = {}
        for idx, v in enumerate(self.training_samples_stack):
            feed_dict['{}training_samples_{}'.format(scope, idx)] = v
        for idx, v in enumerate(self.y_stack):
            feed_dict['{}y_{}'.format(scope, idx)] = v
        for idx, v in enumerate(self.sample_weights_stack):
            feed_dict['{}sample_weights_{}'.format(scope, idx)] = v
        return feed_dict

    def __call__(self, x: TensorList, scope=''):
        """
        Compute residuals
        :param x: [filters, projection_matrices]
        :return: [data_terms, filter_regularizations, proj_mat_regularizations]
        """
        training_samples, y, samples_weights = self.get_inputs(scope)

        filter = x[:len(x) // 2]  # w2 in paper
        P = x[len(x) // 2:]  # w1 in paper

        # Do first convolution
        compressed_samples = operation.conv1x1(
            training_samples, P).apply(self.projection_activation)

        # Do second convolution
        residuals = operation.conv2d(
            compressed_samples, filter,
            mode='same').apply(self.response_activation)

        # Compute data residuals
        residuals = residuals - y

        residuals = residuals * samples_weights.sqrt()

        # Add regularization for projection matrix
        # TODO: remove static_identity
        # for now, this is needed. Otherwise the gradient is None
        residuals.extend(
            filter.apply(static_identity) * self.filter_reg.apply(math.sqrt))

        # Add regularization for projection matrix
        residuals.extend(
            P.apply(static_identity) * self.projection_reg.apply(math.sqrt))

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        # return a.reshape(-1) @ b.reshape(-1)
        num = len(a) // 2  # Number of filters
        a_filter = a[:num]
        b_filter = b[:num]
        a_P = a[num:]
        b_P = b[num:]

        # Filter inner product
        ip_out = a_filter.reshape(-1) @b_filter.reshape(-1)
        # ip_out = operation.conv2d(a_filter, b_filter).view(-1)

        # Add projection matrix part
        ip_out += a_P.reshape(-1) @b_P.reshape(-1)
        # ip_out += operation.conv2d(a_P.view(1, -1, 1, 1), b_P.view(1, -1, 1, 1)).view(-1)

        # Have independent inner products for each filter
        return ip_out.concat(ip_out.clone())

    def M1(self, x: TensorList):
        return x / self.diag_M


class ConvProblem(optimization.L2Problem):
    def __init__(self,
                 training_samples: TensorList,
                 y: TensorList,
                 filter_reg: TensorList,
                 sample_weights: TensorList,
                 response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.response_activation = response_activation

        self.inputs_dict = {}
        # stack tensors
        self.training_samples_stack = None
        self.y_stack = None
        self.sample_weights_stack = None

    def get_feed_dict(self, scope=''):
        if self.training_samples_stack is None or self.y_stack is None or self.sample_weights_stack is None:
            self.training_samples_stack = self.training_samples.apply(
                stack_input)
            self.y_stack = self.y.apply(stack_input)
            self.sample_weights_stack = self.sample_weights.apply(stack_input)
        feed_dict = {}
        for idx, v in enumerate(self.training_samples_stack):
            feed_dict['{}training_samples_{}'.format(scope, idx)] = v
        for idx, v in enumerate(self.y_stack):
            feed_dict['{}y_{}'.format(scope, idx)] = v
        for idx, v in enumerate(self.sample_weights_stack):
            feed_dict['{}sample_weights_{}'.format(scope, idx)] = v
        return feed_dict

    def get_inputs(self, scope=''):
        if scope not in self.inputs_dict:
            training_samples_p = TensorList([
                fluid.layers.data(
                    '{}training_samples_{}'.format(scope, idx),
                    shape=[None] + list(v[0].shape),
                    stop_gradient=False,
                    append_batch_size=False)
                for idx, v in enumerate(self.training_samples)
            ])
            y_p = TensorList([
                fluid.layers.data(
                    '{}y_{}'.format(scope, idx),
                    shape=[None] + list(v[0].shape),
                    stop_gradient=False,
                    append_batch_size=False) for idx, v in enumerate(self.y)
            ])
            sample_weights_p = TensorList([
                fluid.layers.data(
                    '{}sample_weights_{}'.format(scope, idx),
                    shape=[None] + list(v[0].shape),
                    stop_gradient=False,
                    append_batch_size=False)
                for idx, v in enumerate(self.sample_weights)
            ])
            self.inputs_dict[scope] = (training_samples_p, y_p,
                                       sample_weights_p)

        return self.inputs_dict[scope]

    def __call__(self, x: TensorList, scope=''):
        """
        Compute residuals
        :param x: [filters]
        :return: [data_terms, filter_regularizations]
        """
        training_samples, y, samples_weights = self.get_inputs(scope)
        # Do convolution and compute residuals
        residuals = operation.conv2d(
            training_samples, x, mode='same').apply(self.response_activation)
        residuals = residuals - y

        residuals = residuals * samples_weights.sqrt()

        # Add regularization for projection matrix
        residuals.extend(
            x.apply(static_identity) * self.filter_reg.apply(math.sqrt))

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        return a.reshape(-1) @b.reshape(-1)
        # return (a * b).sum()
        # return operation.conv2d(a, b).view(-1)
