#-*- coding: utf-8 -*-
#File: DQN.py
#Author: yobobobo(zhouboacmer@qq.com)

from agent import Model
from atari import AtariPlayer
import gym
import argparse
import cv2
from tqdm import tqdm
from expreplay import ReplayMemory, Experience
import numpy as np
import os
from tensorpack.utils import logger
from atari_wrapper import FrameStack, MapState, FireResetEnv, LimitLength
from collections import deque
from utils import Summary

HIST_LEN = 4
UPDATE_FREQ = 4

#MEMORY_WARMUP_SIZE = 2000
MEMORY_SIZE = 1e6
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

def run_train_episode(agent, env, exp):
    total_reward = 0
    state = env.reset()
    step = 0
    while True:
        step += 1
        history = exp.recent_state()
        history.append(state)
        history = np.stack(history, axis=2)
        action = agent.act(history, train_or_test='train')
        next_state, reward, isOver, _ = env.step(action)
        if reward > 1:
            logger.info("reward:{}".format(reward))
        exp.append(Experience(state, action, reward, isOver))
        # train model
        # start training 
        if len(exp) > MEMORY_WARMUP_SIZE:
            if step % UPDATE_FREQ == 0:
                batch_all_state, batch_action, batch_reward, batch_isOver = exp.sample_batch(args.batch_size)
                batch_state = batch_all_state[:,:,:,:HIST_LEN]
                batch_next_state = batch_all_state[:,:,:,1:]
                #logger.info("batch_state:{}   batch_next_state:{}".format(batch_state.shape, batch_next_state.shape))
                agent.train(batch_state, batch_action, batch_reward, batch_next_state, batch_isOver)
        total_reward += reward
        state = next_state
        if isOver:
            break
    return total_reward, step

def get_player(viz=False, train=False):
    env = AtariPlayer(args.rom, frame_skip=ACTION_REPEAT, viz=viz,
                      live_lost_as_eoe=train, max_num_frames=60000)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    if not train:
        # in training, history is taken care of in expreplay buffer
        env = FrameStack(env, FRAME_HISTORY)
    return env

def eval_agent(agent, env):
    episode_reward = []
    for _ in tqdm(xrange(30), desc='eval agent'):
        state = env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1
            action = agent.act(state, train_or_test='test')
            state, reward, isOver, info = env.step(action)
            total_reward += reward
            if isOver:
                #logger.info("info:{}".format(info['ale.lives']))
                break
        episode_reward.append(total_reward)
    eval_reward = np.mean(episode_reward)
    return eval_reward

def train_agent():
    logger.info("build player")
    env = get_player(train=True)
    test_env = get_player()
    logger.info("build player [end]")
    exp = ReplayMemory(args.mem_size, IMAGE_SIZE, HIST_LEN)
    action_dim = env.action_space.n
    agent = Model(IMAGE_SIZE, action_dim, args.gamma, HIST_LEN, args.use_cuda)

    logger.info("fill ReplayMemory before training")
    with tqdm(total=MEMORY_WARMUP_SIZE) as pbar:
        while len(exp) < MEMORY_WARMUP_SIZE:
            total_reward, step = run_train_episode(agent, env, exp)
            pbar.update(step)

    logger.info("start training, action_dim:{}".format(action_dim))
    
    # train
    test_flag = 0
    summary = Summary(logger.get_logger_dir())
    logger.info("dir:{}".format(logger.get_logger_dir()))
    pbar = tqdm(total=1e8)
    recent_100_reward = []
    total_step = 0
    while True:
        # start epoch
        total_reward, step = run_train_episode(agent, env, exp)
        total_step += step
        pbar.set_description('[train]exploration:{}'.format(agent.exploration))
        pbar.update(step)

        if total_step // 100000 == test_flag:
            pbar.write("testing")
            eval_reward = eval_agent(agent, test_env)
            test_flag += 1
            summary.log_scalar('eval_reward', eval_reward, total_step)
            logger.info("eval_agent done, eval_reward:{}".format(eval_reward))
    pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true',
                        help='if set, use cuda')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for accumulated reward computation')
    parser.add_argument('--mem_size', type=int, default=1000000,
                        help='memory size for experience replay')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--rom', help='atari rom', required=True)
    args = parser.parse_args()

    logger.set_logger_dir(os.path.join('train_log', 'DQN-{}'.format(
        os.path.basename(args.rom).split('.')[0])), action='d')
    train_agent()
