import argparse
import gym
import numpy as np
from itertools import count
import math
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
import paddle.fluid.framework as framework

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor (default: 0.99)')
parser.add_argument(
    '--seed',
    type=int,
    default=543,
    metavar='N',
    help='random seed (default: 543)')
parser.add_argument(
    '--render', action='store_true', help='render the environment')
parser.add_argument('--save_dir', type=str, default="./saved_models")
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)


class Policy(fluid.dygraph.Layer):
    def __init__(self):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)
        self.dropout_ratio = 0.6

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = fluid.layers.reshape(x, shape=[1, 4])
        x = self.affine1(x)
        x = fluid.layers.dropout(x, self.dropout_ratio)
        x = fluid.layers.relu(x)
        action_scores = self.affine2(x)

        self._x_for_debug = x

        return fluid.layers.softmax(action_scores, axis=1)


with fluid.dygraph.guard():
    fluid.default_startup_program().random_seed = args.seed
    fluid.default_main_program().random_seed = args.seed
    np.random.seed(args.seed)

    policy = Policy()

    eps = np.finfo(np.float32).eps.item()
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-2, parameter_list=policy.parameters())

    def get_mean_and_std(values=[]):
        n = 0.
        s = 0.
        for val in values:
            s += val
            n += 1
        mean = s / n

        std = 0.
        for val in values:
            std += (val - mean) * (val - mean)
        std /= n
        std = math.sqrt(std)

        return mean, std

    def sample_action(probs):
        sample = np.random.random()
        idx = 0

        while idx < len(probs) and sample > probs[idx]:
            sample -= probs[idx]
            idx += 1
        mask = [0.] * len(probs)
        mask[idx] = 1.

        return idx, np.array([mask]).astype("float32")

    def choose_best_action(probs):
        idx = 0 if probs[0] > probs[1] else 1
        mask = [1., 0.] if idx == 0 else [0., 1.]

        return idx, np.array([mask]).astype("float32")

    def select_action(state):
        state = fluid.dygraph.base.to_variable(state)
        state.stop_gradient = True
        loss_probs = policy(state)
        probs = loss_probs.numpy()

        action, _mask = sample_action(probs[0])

        mask = fluid.dygraph.base.to_variable(_mask)
        mask.stop_gradient = True

        loss_probs = fluid.layers.log(loss_probs)
        loss_probs = fluid.layers.elementwise_mul(loss_probs, mask)
        loss_probs = fluid.layers.reduce_sum(loss_probs, dim=-1)

        policy.saved_log_probs.append(loss_probs)

        return action

    def finish_episode():
        R = 0
        policy_loss = []
        returns = []
        for r in policy.rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)

        mean, std = get_mean_and_std(returns)

        returns = np.array(returns).astype("float32")
        returns = (returns - mean) / (std + eps)

        for log_prob, R in zip(policy.saved_log_probs, returns):
            log_prob_numpy = log_prob.numpy()

            R_numpy = np.ones_like(log_prob_numpy).astype("float32")
            _R = -1 * R * R_numpy
            _R = fluid.dygraph.base.to_variable(_R)
            _R.stop_gradient = True
            curr_loss = fluid.layers.elementwise_mul(_R, log_prob)
            policy_loss.append(curr_loss)

        policy_loss = fluid.layers.concat(policy_loss)
        policy_loss = fluid.layers.reduce_sum(policy_loss)

        policy_loss.backward()
        optimizer.minimize(policy_loss)

        dy_grad = policy._x_for_debug.gradient()

        policy.clear_gradients()
        del policy.rewards[:]
        del policy.saved_log_probs[:]

        return returns

    running_reward = 10
    state, ep_reward = env.reset(), 0
    model_dict, _ = fluid.load_dygraph(args.save_dir)
    policy.set_dict(model_dict)

    for t in range(1, 10000):  # Don't infinite loop while learning
        state = np.array(state).astype("float32")
        action = select_action(state)
        state, reward, done, _ = env.step(action)

        if args.render:
            env.render()

        policy.rewards.append(reward)
        ep_reward += reward

        if done:
            break

    print('Test reward: {:.2f}'.format(ep_reward))
