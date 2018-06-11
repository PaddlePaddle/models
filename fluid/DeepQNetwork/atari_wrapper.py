# -*- coding: utf-8 -*-

import numpy as np
from collections import deque

import gym
from gym import spaces

_v0, _v1 = gym.__version__.split('.')[:2]
assert int(_v0) > 0 or int(_v1) >= 10, gym.__version__
"""
The following wrappers are copied or modified from openai/baselines:
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""


class MapState(gym.ObservationWrapper):
    def __init__(self, env, map_func):
        gym.ObservationWrapper.__init__(self, env)
        self._func = map_func

    def observation(self, obs):
        return self._func(obs)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        chan = 1 if len(shp) == 2 else shp[2]
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(shp[0], shp[1], chan * k),
                                            dtype=np.uint8)

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k - 1):
            self.frames.append(np.zeros_like(ob))
        self.frames.append(ob)
        return self.observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self.observation(), reward, done, info

    def observation(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=0)


class _FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)


def FireResetEnv(env):
    if isinstance(env, gym.Wrapper):
        baseenv = env.unwrapped
    else:
        baseenv = env
    if 'FIRE' in baseenv.get_action_meanings():
        return _FireResetEnv(env)
    return env


class LimitLength(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k

    def reset(self):
        # This assumes that reset() will really reset the env.
        # If the underlying env tries to be smart about reset
        # (e.g. end-of-life), the assumption doesn't hold.
        ob = self.env.reset()
        self.cnt = 0
        return ob

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        self.cnt += 1
        if self.cnt == self.k:
            done = True
        return ob, r, done, info
