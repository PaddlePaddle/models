#-*- coding: utf-8 -*-
#File: expreplay.py
#Author: yobobobo(zhouboacmer@qq.com)

from tensorpack.utils import logger
from collections import namedtuple
import numpy as np

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'isOver'])

class ReplayMemory(object):
  def __init__(self, max_size, state_shape):
    self.max_size = int(max_size)
    self.state_shape = state_shape

    self.state = np.zeros((self.max_size,) + state_shape, dtype='float32')
    self.action = np.zeros((self.max_size,), dtype='int32')
    self.reward = np.zeros((self.max_size,), dtype='float32')
    self.isOver = np.zeros((self.max_size,), dtype='bool')

    self._curr_size = 0
    self._curr_pos = 0

  def append(self, exp):
    if self._curr_size < self.max_size:
      self._assign(self._curr_pos, exp)
      self._curr_size += 1
    else:
      self._assign(self._curr_pos, exp)
    self._curr_pos = (self._curr_pos + 1) % self.max_size

  def _assign(self, pos, exp):
    self.state[pos] = exp.state
    self.action[pos] = exp.action
    self.reward[pos] = exp.reward
    self.isOver[pos] = exp.isOver

  def __len__(self):
    return self._curr_size

  def sample(self, batch_idx):
    for i, idx in enumerate(batch_idx):
      while (idx + 1) % self._curr_size == self._curr_pos:
        idx = np.random.randint(self._curr_size)
      batch_idx[i] = idx

    next_idx = (batch_idx + 1) % self._curr_size
    state = self.state[batch_idx]
    reward = self.reward[batch_idx]
    action = self.action[batch_idx]
    next_state = self.state[next_idx]
    isOver = self.isOver[batch_idx]
    return (state, action, reward, next_state, isOver)
