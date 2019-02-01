# -*- coding: utf-8 -*-

import numpy as np
import copy
from collections import deque, namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'isOver'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape, context_len):
        self.max_size = int(max_size)
        self.state_shape = state_shape
        self.context_len = int(context_len)

        self.state = np.zeros((self.max_size, ) + state_shape, dtype='uint8')
        self.action = np.zeros((self.max_size, ), dtype='int32')
        self.reward = np.zeros((self.max_size, ), dtype='float32')
        self.isOver = np.zeros((self.max_size, ), dtype='bool')

        self._curr_size = 0
        self._curr_pos = 0
        self._context = deque(maxlen=context_len - 1)

    def append(self, exp):
        """append a new experience into replay memory
        """
        if self._curr_size < self.max_size:
            self._assign(self._curr_pos, exp)
            self._curr_size += 1
        else:
            self._assign(self._curr_pos, exp)
        self._curr_pos = (self._curr_pos + 1) % self.max_size
        if exp.isOver:
            self._context.clear()
        else:
            self._context.append(exp)

    def recent_state(self):
        """ maintain recent state for training"""
        lst = list(self._context)
        states = [np.zeros(self.state_shape, dtype='uint8')] * \
                    (self._context.maxlen - len(lst))
        states.extend([k.state for k in lst])
        return states

    def sample(self, idx):
        """ return state, action, reward, isOver,
            note that some frames in state may be generated from last episode,
            they should be removed from state
            """
        state = np.zeros(
            (self.context_len + 1, ) + self.state_shape, dtype=np.uint8)
        state_idx = np.arange(idx, idx + self.context_len + 1) % self._curr_size

        # confirm that no frame was generated from last episode
        has_last_episode = False
        for k in range(self.context_len - 2, -1, -1):
            to_check_idx = state_idx[k]
            if self.isOver[to_check_idx]:
                has_last_episode = True
                state_idx = state_idx[k + 1:]
                state[k + 1:] = self.state[state_idx]
                break

        if not has_last_episode:
            state = self.state[state_idx]

        real_idx = (idx + self.context_len - 1) % self._curr_size
        action = self.action[real_idx]
        reward = self.reward[real_idx]
        isOver = self.isOver[real_idx]
        return state, reward, action, isOver

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        self.state[pos] = exp.state
        self.reward[pos] = exp.reward
        self.action[pos] = exp.action
        self.isOver[pos] = exp.isOver

    def sample_batch(self, batch_size):
        """sample a batch from replay memory for training
        """
        batch_idx = np.random.randint(
            self._curr_size - self.context_len - 1, size=batch_size)
        batch_idx = (self._curr_pos + batch_idx) % self._curr_size
        batch_exp = [self.sample(i) for i in batch_idx]
        return self._process_batch(batch_exp)

    def _process_batch(self, batch_exp):
        state = np.asarray([e[0] for e in batch_exp], dtype='uint8')
        reward = np.asarray([e[1] for e in batch_exp], dtype='float32')
        action = np.asarray([e[2] for e in batch_exp], dtype='int8')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        return [state, action, reward, isOver]
