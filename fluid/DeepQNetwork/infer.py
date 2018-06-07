#!/usr/bin/env python
# coding=utf8
# File: infer.py

import argparse
import os
import paddle.fluid as fluid
import numpy as np

from tqdm import tqdm
from DQN import get_player

def predict_action(exe, state, predict_program, feed_names, fetch_targets, action_dim):
    if np.random.randint(100) == 0:
        act = np.random.randint(action_dim)
    else:
        state = np.expand_dims(state, axis=0)
        state = np.transpose(state, [0, 3, 1, 2])
        pred_Q = exe.run(predict_program,
                feed={feed_names[0]: state.astype('float32')}, 
                fetch_list=fetch_targets)[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)
    return act
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true',
                        help='if set, use cuda')
    parser.add_argument('--rom', help='atari rom', required=True)
    parser.add_argument('--model_dirname', type=str, default='saved_model',
                        help='dirname to load model')
    parser.add_argument('--model_step_num', type=int, required=True,
                        help='step number of loading model')
    args = parser.parse_args()

    env = get_player(args.rom)
    
    model_path = os.path.join(args.model_dirname, 'DQN-{}'.format(os.path.basename(args.rom).split('.')[0]),
            'step{}'.format(args.model_step_num))
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [predict_program, feed_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        
        episode_reward = [] 
        for _ in tqdm(xrange(30), desc='eval agent'):
            state = env.reset()
            total_reward = 0
            step = 0
            while True:
                step += 1
                action = predict_action(exe, state, predict_program, feed_names, fetch_targets, env.action_space.n)
                state, reward, isOver, info = env.step(action)
                total_reward += reward
                if isOver:
                    break
            episode_reward.append(total_reward)
        eval_reward = np.mean(episode_reward)
        print('Average reward: {}'.format(eval_reward))

            

