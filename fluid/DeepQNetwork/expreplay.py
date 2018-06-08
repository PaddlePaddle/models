# -*- coding: utf-8 -*-
# File: expreplay.py
# Author: Yuxin Wu

import numpy as np
import copy
from collections import deque, namedtuple
import threading
from six.moves import queue, range

from tensorpack.utils import logger

__all__ = ['ExpReplay']

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'isOver'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape, history_len):
        self.max_size = int(max_size)
        self.state_shape = state_shape
        self.history_len = int(history_len)

        self.state = np.zeros((self.max_size,) + state_shape, dtype='uint8')
        self.action = np.zeros((self.max_size,), dtype='int32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.isOver = np.zeros((self.max_size,), dtype='bool')

        self._curr_size = 0
        self._curr_pos = 0
        self._hist = deque(maxlen=history_len - 1)

    def append(self, exp):
        """
        Args:
            exp (Experience):
        """
        if self._curr_size < self.max_size:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
            self._curr_size += 1
        else:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
        if exp.isOver:
            self._hist.clear()
        else:
            self._hist.append(exp)

    def recent_state(self):
        """ return a list of (hist_len-1,) + STATE_SIZE """
        lst = list(self._hist)
        states = [np.zeros(self.state_shape, dtype='uint8')] * (self._hist.maxlen - len(lst))
        states.extend([k.state for k in lst])
        return states

    def sample(self, idx):
        """ return a tuple of (s,r,a,o),
            where s is of shape STATE_SIZE + (hist_len+1,)"""
        state = np.zeros((self.history_len,) + self.state_shape, dtype=np.uint8)
        cur_idx = idx
        for k in range(history_len):
            cur_idx = (idx + k) % self._curr_size
            state = self.state[cur_idx]
            if isOver[cur_idx]:
                break

        action = self.action[cur_idx]
        reward = self.reward[cur_idx]
        isOver = self.isOver[cur_idx]
        state = state.transpose(1, 2, 0)
        return state, reward, action, isOver

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        self.state[pos] = exp.state
        self.reward[pos] = exp.reward
        self.action[pos] = exp.action
        self.isOver[pos] = exp.isOver

    def sample_batch(self, batch_size):
        batch_idx = np.random.randint(self._curr_size - self.history_len - 1, size=batch_size)
        batch_idx = (self._curr_pos + batch_idx) % self._curr_size
        batch_exp = [self.sample(i) for i in batch_idx]
        return self._process_batch(batch_exp)

    def _process_batch(self, batch_exp):
        state = np.asarray([e[0] for e in batch_exp], dtype='uint8')
        reward = np.asarray([e[1] for e in batch_exp], dtype='float32')
        action = np.asarray([e[2] for e in batch_exp], dtype='int8')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        return [state, action, reward, isOver]
