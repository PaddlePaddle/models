#-*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import paddle.fluid as fluid

from train import get_player
from tqdm import tqdm


def predict_action(exe, state, predict_program, feed_names, fetch_targets,
                   action_dim):
    if np.random.random() < 0.01:
        act = np.random.randint(action_dim)
    else:
        state = np.expand_dims(state, axis=0)
        pred_Q = exe.run(predict_program,
                         feed={feed_names[0]: state.astype('float32')},
                         fetch_list=fetch_targets)[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)
    return act


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_cuda', action='store_true', help='if set, use cuda')
    parser.add_argument('--rom', type=str, required=True, help='atari rom')
    parser.add_argument(
        '--model_path', type=str, required=True, help='dirname to load model')
    parser.add_argument(
        '--viz',
        type=float,
        default=0,
        help='''viz: visualization setting:
                Set to 0 to disable;
                Set to a positive number to be the delay between frames to show.
             ''')
    args = parser.parse_args()

    env = get_player(args.rom, viz=args.viz)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.Scope()
    with fluid.scope_guard(inference_scope):
        [predict_program, feed_names,
         fetch_targets] = fluid.io.load_inference_model(args.model_path, exe)

        episode_reward = []
        for _ in tqdm(xrange(30), desc='eval agent'):
            state = env.reset()
            total_reward = 0
            while True:
                action = predict_action(exe, state, predict_program, feed_names,
                                        fetch_targets, env.action_space.n)
                state, reward, isOver, info = env.step(action)
                total_reward += reward
                if isOver:
                    break
            episode_reward.append(total_reward)
        eval_reward = np.mean(episode_reward)
        print('Average reward of 30 epidose: {}'.format(eval_reward))
