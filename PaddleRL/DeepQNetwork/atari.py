# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import threading

import gym
from gym import spaces
from gym.envs.atari.atari_env import ACTION_MEANING

from atari_py import ALEInterface

__all__ = ['AtariPlayer']

ROM_URL = "https://github.com/openai/atari-py/tree/master/atari_py/atari_roms"
_ALE_LOCK = threading.Lock()
"""
The following AtariPlayer are copied or modified from tensorpack/tensorpack:
    https://github.com/tensorpack/tensorpack/blob/master/examples/DeepQNetwork/atari.py
"""


class AtariPlayer(gym.Env):
    """
    A wrapper for ALE emulator, with configurations to mimic DeepMind DQN settings.
    Info:
        score: the accumulated reward in the current game
        gameOver: True when the current game is Over
    """

    def __init__(self,
                 rom_file,
                 viz=0,
                 frame_skip=4,
                 nullop_start=30,
                 live_lost_as_eoe=True,
                 max_num_frames=0):
        """
        Args:
            rom_file: path to the rom
            frame_skip: skip every k frames and repeat the action
            viz: visualization to be done.
                Set to 0 to disable.
                Set to a positive number to be the delay between frames to show.
                Set to a string to be a directory to store frames.
            nullop_start: start with random number of null ops.
            live_losts_as_eoe: consider lost of lives as end of episode. Useful for training.
            max_num_frames: maximum number of frames per episode.
        """
        super(AtariPlayer, self).__init__()
        assert os.path.isfile(rom_file), \
            "rom {} not found. Please download at {}".format(rom_file, ROM_URL)

        try:
            ALEInterface.setLoggerMode(ALEInterface.Logger.Error)
        except AttributeError:
            print("You're not using latest ALE")

        # avoid simulator bugs: https://github.com/mgbellemare/Arcade-Learning-Environment/issues/86
        with _ALE_LOCK:
            self.ale = ALEInterface()
            self.ale.setInt(b"random_seed", np.random.randint(0, 30000))
            self.ale.setInt(b"max_num_frames_per_episode", max_num_frames)
            self.ale.setBool(b"showinfo", False)

            self.ale.setInt(b"frame_skip", 1)
            self.ale.setBool(b'color_averaging', False)
            # manual.pdf suggests otherwise.
            self.ale.setFloat(b'repeat_action_probability', 0.0)

            # viz setup
            if isinstance(viz, str):
                assert os.path.isdir(viz), viz
                self.ale.setString(b'record_screen_dir', viz)
                viz = 0
            if isinstance(viz, int):
                viz = float(viz)
            self.viz = viz
            if self.viz and isinstance(self.viz, float):
                self.windowname = os.path.basename(rom_file)
                cv2.startWindowThread()
                cv2.namedWindow(self.windowname)

            self.ale.loadROM(rom_file.encode('utf-8'))
        self.width, self.height = self.ale.getScreenDims()
        self.actions = self.ale.getMinimalActionSet()

        self.live_lost_as_eoe = live_lost_as_eoe
        self.frame_skip = frame_skip
        self.nullop_start = nullop_start

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.height, self.width),
                                            dtype=np.uint8)
        self._restart_episode()

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self.actions]

    def _grab_raw_image(self):
        """
        :returns: the current 3-channel image
        """
        m = self.ale.getScreenRGB()
        return m.reshape((self.height, self.width, 3))

    def _current_state(self):
        """
        returns: a gray-scale (h, w) uint8 image
        """
        ret = self._grab_raw_image()
        # avoid missing frame issue: max-pooled over the last screen
        ret = np.maximum(ret, self.last_raw_screen)
        if self.viz:
            if isinstance(self.viz, float):
                cv2.imshow(self.windowname, ret)
                cv2.waitKey(int(self.viz * 1000))
        ret = ret.astype('float32')
        # 0.299,0.587.0.114. same as rgb2y in torch/image
        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        return ret.astype('uint8')  # to save some memory

    def _restart_episode(self):
        with _ALE_LOCK:
            self.ale.reset_game()

        # random null-ops start
        n = np.random.randint(self.nullop_start)
        self.last_raw_screen = self._grab_raw_image()
        for k in range(n):
            if k == n - 1:
                self.last_raw_screen = self._grab_raw_image()
            self.ale.act(0)

    def reset(self):
        if self.ale.game_over():
            self._restart_episode()
        return self._current_state()

    def step(self, act):
        oldlives = self.ale.lives()
        r = 0
        for k in range(self.frame_skip):
            if k == self.frame_skip - 1:
                self.last_raw_screen = self._grab_raw_image()
            r += self.ale.act(self.actions[act])
            newlives = self.ale.lives()
            if self.ale.game_over() or \
                    (self.live_lost_as_eoe and newlives < oldlives):
                break

        isOver = self.ale.game_over()
        if self.live_lost_as_eoe:
            isOver = isOver or newlives < oldlives

        info = {'ale.lives': newlives}
        return self._current_state(), r, isOver, info
