#-*- coding: utf-8 -*-

from DQN_agent import DQNModel
from DoubleDQN_agent import DoubleDQNModel
from DuelingDQN_agent import DuelingDQNModel
from atari import AtariPlayer
import paddle.fluid as fluid
import gym
import argparse
import cv2
from tqdm import tqdm
from expreplay import ReplayMemory, Experience
import numpy as np
import os

from datetime import datetime
from atari_wrapper import FrameStack, MapState, FireResetEnv, LimitLength
from collections import deque

UPDATE_FREQ = 4

MEMORY_SIZE = 1e6
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20
IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4
ACTION_REPEAT = 4  # aka FRAME_SKIP
UPDATE_FREQ = 4


def run_train_episode(agent, env, exp):
    total_reward = 0
    state = env.reset()
    step = 0
    while True:
        step += 1
        context = exp.recent_state()
        context.append(state)
        context = np.stack(context, axis=0)
        action = agent.act(context, train_or_test='train')
        next_state, reward, isOver, _ = env.step(action)
        exp.append(Experience(state, action, reward, isOver))
        # train model
        # start training 
        if len(exp) > MEMORY_WARMUP_SIZE:
            if step % UPDATE_FREQ == 0:
                batch_all_state, batch_action, batch_reward, batch_isOver = exp.sample_batch(
                    args.batch_size)
                batch_state = batch_all_state[:, :CONTEXT_LEN, :, :]
                batch_next_state = batch_all_state[:, 1:, :, :]
                agent.train(batch_state, batch_action, batch_reward,
                            batch_next_state, batch_isOver)
        total_reward += reward
        state = next_state
        if isOver:
            break
    return total_reward, step


def get_player(rom, viz=False, train=False):
    env = AtariPlayer(
        rom,
        frame_skip=ACTION_REPEAT,
        viz=viz,
        live_lost_as_eoe=train,
        max_num_frames=60000)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    if not train:
        # in training, context is taken care of in expreplay buffer
        env = FrameStack(env, CONTEXT_LEN)
    return env


def eval_agent(agent, env):
    episode_reward = []
    for _ in tqdm(range(30), desc='eval agent'):
        state = env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1
            action = agent.act(state, train_or_test='test')
            state, reward, isOver, info = env.step(action)
            total_reward += reward
            if isOver:
                break
        episode_reward.append(total_reward)
    eval_reward = np.mean(episode_reward)
    return eval_reward


def train_agent():
    env = get_player(args.rom, train=True)
    test_env = get_player(args.rom)
    exp = ReplayMemory(args.mem_size, IMAGE_SIZE, CONTEXT_LEN)
    action_dim = env.action_space.n

    if args.alg == 'DQN':
        agent = DQNModel(IMAGE_SIZE, action_dim, args.gamma, CONTEXT_LEN,
                         args.use_cuda)
    elif args.alg == 'DoubleDQN':
        agent = DoubleDQNModel(IMAGE_SIZE, action_dim, args.gamma, CONTEXT_LEN,
                               args.use_cuda)
    elif args.alg == 'DuelingDQN':
        agent = DuelingDQNModel(IMAGE_SIZE, action_dim, args.gamma, CONTEXT_LEN,
                                args.use_cuda)
    else:
        print('Input algorithm name error!')
        return

    with tqdm(total=MEMORY_WARMUP_SIZE, desc='Memory warmup') as pbar:
        while len(exp) < MEMORY_WARMUP_SIZE:
            total_reward, step = run_train_episode(agent, env, exp)
            pbar.update(step)

    # train
    test_flag = 0
    save_flag = 0
    pbar = tqdm(total=1e8)
    recent_100_reward = []
    total_step = 0
    max_reward = None
    save_path = os.path.join(args.model_dirname, '{}-{}'.format(
        args.alg, os.path.basename(args.rom).split('.')[0]))
    while True:
        # start epoch
        total_reward, step = run_train_episode(agent, env, exp)
        total_step += step
        pbar.set_description('[train]exploration:{}'.format(agent.exploration))
        pbar.update(step)

        if total_step // args.test_every_steps == test_flag:
            pbar.write("testing")
            eval_reward = eval_agent(agent, test_env)
            test_flag += 1
            print("eval_agent done, (steps, eval_reward): ({}, {})".format(
                total_step, eval_reward))

            if max_reward is None or eval_reward > max_reward:
                max_reward = eval_reward
                fluid.io.save_inference_model(save_path, ['state'],
                                              agent.pred_value, agent.exe,
                                              agent.predict_program)
    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--alg',
        type=str,
        default='DQN',
        help='Reinforcement learning algorithm, support: DQN, DoubleDQN, DuelingDQN'
    )
    parser.add_argument(
        '--use_cuda', action='store_true', help='if set, use cuda')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for accumulated reward computation')
    parser.add_argument(
        '--mem_size',
        type=int,
        default=1000000,
        help='memory size for experience replay')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--rom', help='atari rom', required=True)
    parser.add_argument(
        '--model_dirname',
        type=str,
        default='saved_model',
        help='dirname to save model')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=100000,
        help='every steps number to run test')
    args = parser.parse_args()
    train_agent()
