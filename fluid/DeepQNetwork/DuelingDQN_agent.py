#-*- coding: utf-8 -*-

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import numpy as np
from tqdm import tqdm
import math
from utils import fluid_flatten


class DuelingDQNModel(object):
    def __init__(self, state_dim, action_dim, gamma, hist_len, use_cuda=False):
        self.img_height = state_dim[0]
        self.img_width = state_dim[1]
        self.action_dim = action_dim
        self.gamma = gamma
        self.exploration = 1.1
        self.update_target_steps = 10000 // 4
        self.hist_len = hist_len
        self.use_cuda = use_cuda

        self.global_step = 0
        self._build_net()

    def _get_inputs(self):
        return fluid.layers.data(
                   name='state',
                   shape=[self.hist_len, self.img_height, self.img_width],
                   dtype='float32'), \
               fluid.layers.data(
                   name='action', shape=[1], dtype='int32'), \
               fluid.layers.data(
                   name='reward', shape=[], dtype='float32'), \
               fluid.layers.data(
                   name='next_s',
                   shape=[self.hist_len, self.img_height, self.img_width],
                   dtype='float32'), \
               fluid.layers.data(
                   name='isOver', shape=[], dtype='bool')

    def _build_net(self):
        state, action, reward, next_s, isOver = self._get_inputs()
        self.pred_value = self.get_DQN_prediction(state)
        self.predict_program = fluid.default_main_program().clone()

        reward = fluid.layers.clip(reward, min=-1.0, max=1.0)

        action_onehot = fluid.layers.one_hot(action, self.action_dim)
        action_onehot = fluid.layers.cast(action_onehot, dtype='float32')

        pred_action_value = fluid.layers.reduce_sum(
            fluid.layers.elementwise_mul(action_onehot, self.pred_value), dim=1)

        targetQ_predict_value = self.get_DQN_prediction(next_s, target=True)
        best_v = fluid.layers.reduce_max(targetQ_predict_value, dim=1)
        best_v.stop_gradient = True

        target = reward + (1.0 - fluid.layers.cast(
            isOver, dtype='float32')) * self.gamma * best_v
        cost = fluid.layers.square_error_cost(pred_action_value, target)
        cost = fluid.layers.reduce_mean(cost)

        self._sync_program = self._build_sync_target_network()

        optimizer = fluid.optimizer.Adam(1e-3 * 0.5, epsilon=1e-3)
        optimizer.minimize(cost)

        # define program
        self.train_program = fluid.default_main_program()

        # fluid exe
        place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
        self.exe = fluid.Executor(place)
        self.exe.run(fluid.default_startup_program())

    def get_DQN_prediction(self, image, target=False):
        image = image / 255.0

        variable_field = 'target' if target else 'policy'

        conv1 = fluid.layers.conv2d(
            input=image,
            num_filters=32,
            filter_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            act='relu',
            param_attr=ParamAttr(name='{}_conv1'.format(variable_field)),
            bias_attr=ParamAttr(name='{}_conv1_b'.format(variable_field)))
        max_pool1 = fluid.layers.pool2d(
            input=conv1, pool_size=[2, 2], pool_stride=[2, 2], pool_type='max')

        conv2 = fluid.layers.conv2d(
            input=max_pool1,
            num_filters=32,
            filter_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            act='relu',
            param_attr=ParamAttr(name='{}_conv2'.format(variable_field)),
            bias_attr=ParamAttr(name='{}_conv2_b'.format(variable_field)))
        max_pool2 = fluid.layers.pool2d(
            input=conv2, pool_size=[2, 2], pool_stride=[2, 2], pool_type='max')

        conv3 = fluid.layers.conv2d(
            input=max_pool2,
            num_filters=64,
            filter_size=[4, 4],
            stride=[1, 1],
            padding=[1, 1],
            act='relu',
            param_attr=ParamAttr(name='{}_conv3'.format(variable_field)),
            bias_attr=ParamAttr(name='{}_conv3_b'.format(variable_field)))
        max_pool3 = fluid.layers.pool2d(
            input=conv3, pool_size=[2, 2], pool_stride=[2, 2], pool_type='max')

        conv4 = fluid.layers.conv2d(
            input=max_pool3,
            num_filters=64,
            filter_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            act='relu',
            param_attr=ParamAttr(name='{}_conv4'.format(variable_field)),
            bias_attr=ParamAttr(name='{}_conv4_b'.format(variable_field)))

        flatten = fluid_flatten(conv4)

        value = fluid.layers.fc(
            input=flatten,
            size=1,
            param_attr=ParamAttr(name='{}_value_fc'.format(variable_field)),
            bias_attr=ParamAttr(name='{}_value_fc_b'.format(variable_field)))

        advantage = fluid.layers.fc(
            input=flatten,
            size=self.action_dim,
            param_attr=ParamAttr(name='{}_advantage_fc'.format(variable_field)),
            bias_attr=ParamAttr(
                name='{}_advantage_fc_b'.format(variable_field)))

        Q = advantage + (value - fluid.layers.reduce_mean(
            advantage, dim=1, keep_dim=True))
        return Q

    def _build_sync_target_network(self):
        vars = list(fluid.default_main_program().list_vars())
        policy_vars = list(filter(
            lambda x: 'GRAD' not in x.name and 'policy' in x.name, vars))
        target_vars = list(filter(
            lambda x: 'GRAD' not in x.name and 'target' in x.name, vars))
        policy_vars.sort(key=lambda x: x.name)
        target_vars.sort(key=lambda x: x.name)

        sync_program = fluid.default_main_program().clone()
        with fluid.program_guard(sync_program):
            sync_ops = []
            for i, var in enumerate(policy_vars):
                sync_op = fluid.layers.assign(policy_vars[i], target_vars[i])
                sync_ops.append(sync_op)
        # The prune API is deprecated, please don't use it any more.
        sync_program = sync_program._prune(sync_ops)
        return sync_program

    def act(self, state, train_or_test):
        sample = np.random.random()
        if train_or_test == 'train' and sample < self.exploration:
            act = np.random.randint(self.action_dim)
        else:
            if np.random.random() < 0.01:
                act = np.random.randint(self.action_dim)
            else:
                state = np.expand_dims(state, axis=0)
                pred_Q = self.exe.run(self.predict_program,
                                      feed={'state': state.astype('float32')},
                                      fetch_list=[self.pred_value])[0]
                pred_Q = np.squeeze(pred_Q, axis=0)
                act = np.argmax(pred_Q)
        if train_or_test == 'train':
            self.exploration = max(0.1, self.exploration - 1e-6)
        return act

    def train(self, state, action, reward, next_state, isOver):
        if self.global_step % self.update_target_steps == 0:
            self.sync_target_network()
        self.global_step += 1

        action = np.expand_dims(action, -1)
        self.exe.run(self.train_program, \
                  feed={'state': state.astype('float32'), \
                        'action': action.astype('int32'), \
                        'reward': reward, \
                        'next_s': next_state.astype('float32'), \
                        'isOver': isOver})

    def sync_target_network(self):
        self.exe.run(self._sync_program)
