import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
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
parser.add_argument('--save_dir', type=str, default="./saved_models_ac")
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(fluid.dygraph.Layer):
    def __init__(self):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = fluid.layers.reshape(x, shape=[1, 4])
        x = self.affine1(x)
        x = fluid.layers.relu(x)

        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return fluid.layers.softmax(action_scores, axis=-1), state_values


with fluid.dygraph.guard():
    fluid.default_startup_program().random_seed = args.seed
    fluid.default_main_program().random_seed = args.seed
    np.random.seed(args.seed)
    policy = Policy()

    eps = np.finfo(np.float32).eps.item()
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=3e-2, parameter_list=policy.parameters())

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
        probs, state_value = policy(state)
        np_probs = probs.numpy()

        action, _mask = sample_action(np_probs[0])

        mask = fluid.dygraph.base.to_variable(_mask)
        mask.stop_gradient = True

        loss_probs = fluid.layers.log(probs)
        loss_probs = fluid.layers.elementwise_mul(loss_probs, mask)
        loss_probs = fluid.layers.reduce_sum(loss_probs, dim=-1)

        policy.saved_actions.append(SavedAction(loss_probs, state_value))

        return action

    def finish_episode():
        R = 0
        saved_actions = policy.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in policy.rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)

        mean, std = get_mean_and_std(returns)
        returns = np.array(returns).astype("float32")
        returns = (returns - mean) / (std + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value[0][0]

            log_prob_numpy = log_prob.numpy()
            R_numpy = np.ones_like(log_prob_numpy).astype("float32")
            _R = -1 * advantage * R_numpy
            _R = fluid.dygraph.base.to_variable(_R)
            _R.stop_gradient = True

            policy_loss = fluid.layers.elementwise_mul(_R, log_prob)
            policy_losses.append(policy_loss)

            _R2 = np.ones_like(value.numpy()).astype("float32") * R
            _R2 = fluid.dygraph.base.to_variable(_R2)
            _R2.stop_gradient = True

            value_loss = fluid.layers.smooth_l1(value, _R2, sigma=1.0)
            value_losses.append(value_loss)

        all_policy_loss = fluid.layers.concat(policy_losses)
        all_policy_loss = fluid.layers.reduce_sum(all_policy_loss)

        all_value_loss = fluid.layers.concat(value_losses)
        all_value_loss = fluid.layers.reduce_sum(all_value_loss)

        loss = all_policy_loss + all_value_loss

        loss.backward()
        optimizer.minimize(loss)

        policy.clear_gradients()
        del policy.rewards[:]
        del policy.saved_actions[:]

        return returns

    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
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

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        returns = finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.
                  format(i_episode, ep_reward, running_reward))
            #print(returns)
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(
                      running_reward, t))
            fluid.save_dygraph(policy.state_dict(), args.save_dir)
            break
