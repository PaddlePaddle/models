#-*- coding: utf-8 -*-
#File: agent.py
#Author: yobobobo(zhouboacmer@qq.com)

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import numpy as np
from tqdm import tqdm
import math
from rllab.utils import logger
from rllab import lab
from tensorpack.utils.globvars import globalns as param


UPDATE_TARGET_STEPS = 10000 // 4


class Model(object):
    def __init__(self, state_dim, action_dim, gamma):
        self.img_height = state_dim[0]
        self.img_width = state_dim[1]
        self.action_dim = action_dim
        self.gamma = gamma
        self.exploration = 1.1

        self.global_step = 0
        self._build_net()

    def _get_inputs(self):
        return [lab.data(name='state', shape=[param.hist_len, self.img_height, self.img_width], dtype='float32'),
                lab.data(name='action', shape=[1], dtype='int32'), 
                lab.data(name='reward', shape=[], dtype='float32'),
                lab.data(name='next_s', shape=[param.hist_len, self.img_height, self.img_width], dtype='float32'),
                lab.data(name='isOver', shape=[], dtype='bool')]

    def _build_net(self):
        state, action, reward, next_s, isOver = self._get_inputs()
        state = lab.cast(state, 'float32')
        with lab.variable_scope('policy'):
            self.pred_value = self.get_DQN_prediction(state)
        self.predict_program = fluid.default_main_program().clone()

        next_s = lab.cast(next_s, 'float32')
        reward = lab.clip(reward, min=-1.0, max=1.0)

        action_onehot = lab.one_hot(action, self.action_dim)
        action_onehot = lab.cast(action_onehot, dtype='float32')

        pred_action_value = lab.reduce_sum(\
                            lab.elementwise_mul(action_onehot, self.pred_value), dim=1)

        with lab.variable_scope('target'):
            targetQ_predict_value = self.get_DQN_prediction(next_s)
        best_v = lab.reduce_max(targetQ_predict_value, dim=1)
        best_v = lab.StopGradient(best_v)
        #best_v.stop_gradient = True

        target = reward + (1.0 - lab.cast(isOver, dtype='float32')) * self.gamma * best_v
        cost = lab.SquareError(pred_action_value, target)
        cost = lab.reduce_mean(cost)

        self._sync_program = self._build_sync_target_network()

        optimizer = fluid.optimizer.Adam(1e-3 * 0.5, epsilon=1e-3)
        optimizer.minimize(cost)

        # define program
        self.train_program = fluid.default_main_program()

        # fluid exe
        place = fluid.CUDAPlace(0)
        #place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)
        self.exe.run(fluid.default_startup_program())

    def get_DQN_prediction(self, image):
        image = image / 255.0
        l = lab.Conv2D('conv1', image, num_filters=32, filter_size=[5, 5], padding=[2, 2], act='relu')
        l = lab.MaxPooling(l, pool_size=[2, 2], pool_stride=[2, 2])

        l = lab.Conv2D('conv2', l, num_filters=32, filter_size=[5, 5], padding=[2, 2], act='relu')
        l = lab.MaxPooling(l, pool_size=[2, 2], pool_stride=[2, 2])

        l = lab.Conv2D('conv3', l, num_filters=64, filter_size=[4, 4], padding=[1, 1], act='relu')
        l = lab.MaxPooling(l, pool_size=[2, 2], pool_stride=[2, 2])

        l = lab.Conv2D('conv4', l, num_filters=64, filter_size=[3, 3], padding=[1, 1], act='relu')
        logger.info("l:{}".format(l))
        
        l = lab.reshape(l, shape=[-1, 6400])
        value = lab.FullyConnected('value', l, self.action_dim)

        return value

    def _build_sync_target_network(self):
        vars = list(fluid.default_main_program().list_vars())
        policy_vars = filter(lambda x: 'GRAD' not in x.name and 'policy' in x.name, vars)
        target_vars = filter(lambda x: 'GRAD' not in x.name and 'target' in x.name, vars)
        policy_vars.sort(key=lambda x:x.name)
        target_vars.sort(key=lambda x:x.name)

        sync_program = fluid.default_main_program().clone()
        with fluid.program_guard(sync_program):
            sync_ops = []
            for i, var in enumerate(policy_vars):
                logger.info("[assign] policy:{}   target:{}".format(policy_vars[i].name, target_vars[i].name))
                sync_op = lab.assign(policy_vars[i], target_vars[i])
                sync_ops.append(sync_op)
        sync_program = sync_program.prune(sync_ops)
        return sync_program

    def act(self, state, train_or_test):
        sample = np.random.random()
        if train_or_test == 'train' and sample < self.exploration:
            act = np.random.randint(self.action_dim)
        else:
            if np.random.randint(100) == 0:
                act = np.random.randint(self.action_dim)
            else:
                state = np.expand_dims(state, axis=0)
                state = np.transpose(state, [0, 3, 1, 2])
                pred_Q = self.exe.run(self.predict_program,
                                    feed={'state': state.astype('float32')},
                                    fetch_list=[self.pred_value])[0]
                pred_Q = np.squeeze(pred_Q, axis=0)
                act = np.argmax(pred_Q)
        if train_or_test == 'train':
            self.exploration = max(0.1, self.exploration - 1e-6)
        return act

    def train(self, state, action, reward, next_state, isOver):
        if self.global_step % UPDATE_TARGET_STEPS == 0:
            self.sync_target_network()
        self.global_step += 1
        # state -> nchw
        state = np.transpose(state, [0, 3, 1, 2])
        next_state = np.transpose(next_state, [0, 3, 1, 2])

        action = np.expand_dims(action, -1)
        self.exe.run(self.train_program, \
                  feed={'state': state.astype('float32'), \
                        'action': action.astype('int32'), \
                        'reward': reward, \
                        'next_s': next_state.astype('float32'), \
                        'isOver': isOver})

    def sync_target_network(self):
        self.exe.run(self._sync_program)
