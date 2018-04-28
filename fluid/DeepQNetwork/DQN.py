#-*- coding: utf-8 -*-
#File: DQN.py
#Author: yobobobo(zhouboacmer@qq.com)

from agent import Model
import gym
import argparse
from tqdm import tqdm
from expreplay import ReplayMemory, Experience
import numpy as np
import os

UPDATE_FREQ = 4
MEMORY_WARMUP_SIZE = 1000 

def run_episode(agent, env, exp, train_or_test):
  assert train_or_test in ['train', 'test'], train_or_test
  total_reward = 0
  state = env.reset()
  for step in range(200):
    action = agent.act(state, train_or_test)
    next_state, reward, isOver, _ = env.step(action)
    if train_or_test == 'train':
      exp.append(Experience(state, action, reward, isOver))
      # train model
      # start training 
      if len(exp) > MEMORY_WARMUP_SIZE:
        batch_idx = np.random.randint(len(exp), size=(args.batch_size))
        if step % UPDATE_FREQ == 0:
          batch_state, batch_action, batch_reward, batch_next_state, batch_isOver = exp.sample(batch_idx)
          agent.train(batch_state, batch_action, batch_reward, batch_next_state, batch_isOver)
    total_reward += reward
    state = next_state
    if isOver:
      break
  return total_reward

def train_agent():
  env = gym.make(args.env)
  state_shape = env.observation_space.shape
  exp = ReplayMemory(args.mem_size, state_shape)
  action_dim = env.action_space.n
  agent = Model(state_shape[0], action_dim, gamma=0.99)

  while len(exp) < MEMORY_WARMUP_SIZE:
    run_episode(agent, env, exp, train_or_test='train')

  max_episode = 4000
  
  # train
  total_episode = 0
  pbar = tqdm(total=max_episode)
  recent_100_reward = []
  for episode in xrange(max_episode):
    # start epoch
    total_reward = run_episode(agent, env, exp, train_or_test='train')
    pbar.set_description('training, exploration:{}'.format(agent.exploration))
    pbar.update()

    # recent 100 reward
    total_reward = run_episode(agent, env, exp, train_or_test='test')
    recent_100_reward.append(total_reward)
    if len(recent_100_reward) > 100:
      recent_100_reward = recent_100_reward[1:]
    pbar.write("episode:{}  test_reward:{}".format(episode, np.mean(recent_100_reward)))

  pbar.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', type=str, default='MountainCar-v0', help='enviroment to train DQN model, e.g CartPole-v0')
  parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for accumulated reward computation')
  parser.add_argument('--mem_size', type=int, default=500000, help='memory size for experience replay')
  parser.add_argument('--batch_size', type=int, default=192, help='batch size for training')
  args = parser.parse_args()

  train_agent()
