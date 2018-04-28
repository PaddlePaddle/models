#-*- coding: utf-8 -*-
#File: agent.py
#Author: yobobobo(zhouboacmer@qq.com)

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import numpy as np
from tqdm import tqdm
import math

UPDATE_TARGET_STEPS = 200

class Model(object):
    def __init__(self, state_dim, action_dim, gamma):
        self.global_step = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.exploration = 1.0

        self._build_net()

    def _get_inputs(self):
        return [fluid.layers.data(name='state', shape=[self.state_dim], dtype='float32'),
                        fluid.layers.data(name='action', shape=[1], dtype='int32'),
                        fluid.layers.data(name='reward', shape=[], dtype='float32'),
                        fluid.layers.data(name='next_s', shape=[self.state_dim], dtype='float32'),
                        fluid.layers.data(name='isOver', shape=[], dtype='bool')]

    def _build_net(self):
        state, action, reward, next_s, isOver = self._get_inputs()
        self.pred_value = self.get_DQN_prediction(state)
        self.predict_program = fluid.default_main_program().clone()

        action_onehot = fluid.layers.one_hot(action, self.action_dim)
        action_onehot = fluid.layers.cast(action_onehot, dtype='float32')

        pred_action_value = fluid.layers.reduce_sum(\
                    fluid.layers.elementwise_mul(action_onehot, self.pred_value), dim=1)

        targetQ_predict_value = self.get_DQN_prediction(next_s, target=True)
        best_v = fluid.layers.reduce_max(targetQ_predict_value, dim=1)
        best_v.stop_gradient = True

        target = reward + (1.0 - fluid.layers.cast(isOver, dtype='float32')) * self.gamma * best_v
        cost = fluid.layers.square_error_cost(input=pred_action_value, label=target)
        cost = fluid.layers.reduce_mean(cost)

        self._sync_program = self._build_sync_target_network()

        optimizer = fluid.optimizer.Adam(1e-3)
        optimizer.minimize(cost)

        # define program
        self.train_program = fluid.default_main_program()

        # fluid exe
        place = fluid.CUDAPlace(0)
        self.exe = fluid.Executor(place)
        self.exe.run(fluid.default_startup_program())

    def get_DQN_prediction(self, state, target=False):
        variable_field = 'target' if target else 'policy'
        # layer fc1
        param_attr = ParamAttr(name='{}_fc1'.format(variable_field))
        bias_attr = ParamAttr(name='{}_fc1_b'.format(variable_field))
        fc1 = fluid.layers.fc(input=state, 
                              size=256,
                              act='relu',
                              param_attr=param_attr,
                              bias_attr=bias_attr)

        param_attr = ParamAttr(name='{}_fc2'.format(variable_field))
        bias_attr = ParamAttr(name='{}_fc2_b'.format(variable_field))
        fc2 = fluid.layers.fc(input=fc1, 
                              size=128, 
                              act='tanh', 
                              param_attr=param_attr, 
                              bias_attr=bias_attr)

        param_attr = ParamAttr(name='{}_fc3'.format(variable_field))
        bias_attr = ParamAttr(name='{}_fc3_b'.format(variable_field))
        value = fluid.layers.fc(input=fc2, 
                                size=self.action_dim, 
                                param_attr=param_attr, 
                                bias_attr=bias_attr)

        return value

    def _build_sync_target_network(self):
        vars = fluid.default_main_program().list_vars()
        policy_vars = []
        target_vars = []
        for var in vars:
            if 'GRAD' in var.name: continue
            if 'policy' in var.name:
                policy_vars.append(var)
            elif 'target' in var.name:
                target_vars.append(var)

        policy_vars.sort(key=lambda x: x.name.split('policy_')[1])
        target_vars.sort(key=lambda x: x.name.split('target_')[1])

        sync_program = fluid.default_main_program().clone()
        with fluid.program_guard(sync_program):
            sync_ops = []
            for i, var in enumerate(policy_vars):
                sync_op = fluid.layers.assign(policy_vars[i], target_vars[i])
                sync_ops.append(sync_op)
        sync_program = sync_program.prune(sync_ops)
        return sync_program

    def act(self, state, train_or_test):
        sample = np.random.random()
        if train_or_test == 'train' and sample < self.exploration:
            act = np.random.randint(self.action_dim)
        else:
            state = np.expand_dims(state, axis=0)
            pred_Q = self.exe.run(self.predict_program, 
                                feed={'state': state.astype('float32')},
                                fetch_list=[self.pred_value])[0]
            pred_Q = np.squeeze(pred_Q, axis=0)
            act = np.argmax(pred_Q)
        self.exploration = max(0.1, self.exploration - 1e-6)
        return act 

    def train(self, state, action, reward, next_state, isOver):
        if self.global_step % UPDATE_TARGET_STEPS == 0:
            self.sync_target_network()
        self.global_step += 1

        action = np.expand_dims(action, -1)
        self.exe.run(self.train_program, \
                  feed={'state': state, \
                        'action': action, \
                        'reward': reward, \
                        'next_s': next_state, \
                        'isOver': isOver})

    def sync_target_network(self):
        self.exe.run(self._sync_program)
