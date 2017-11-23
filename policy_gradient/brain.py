import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.io import save_persistables, load_persistables
from paddle.v2.fluid.optimizer import SGDOptimizer

# reproducible
np.random.seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False, ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.build_net(self)
        self.place = core.CPUPlace()
        self.exe = Executor(self.place)

    def build_net(self):

        obs = layers.data(
            name='obs', shape=[self.n_features], data_type='float32')
        acts = layers.data(name='acts', shape=[1], data_type='int32')
        vt = layers.data(name='vt', shape=[1], data_type='float32')
        # fc1
        fc1 = layers.fc(
            input=obs,
            size=10,
            act="tanh"  # tanh activation
        )
        # fc2
        all_act_prob = layers.fc(input=fc1, size=self.n_actions, act="softmax")
        # to maximize total reward (log_p * R) is to minimize -(log_p * R)
        neg_log_prob = layers.cross_entropy(
            input=all_act_prob,
            label=acts)  # this is negative log of chosen action
        neg_log_prob_weight = layers.elementwise_mul(x=neg_log_prob, y=vt)
        loss = layers.reduce_mean(x=neg_log_prob_weight)  # reward guided loss

        self.optimizer = SGDOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.exe.run(
            framework.default_main_program().prune(all_act_prob),
            feed={"obs": observation[np.newaxis, :]},
            fetch_list=[all_act_prob])
        prob_weights = np.array(prob_weights[0])
        action = np.random.choice(
            range(prob_weights.shape[1]),
            p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        #print framework.default_main_program()
        tensor_obs = core.LoDTensor()
        tensor_obs.set(np.vstack(self.ep_obs), self.place)
        tensor_as = core.LoDTensor()
        tensor_as.set(np.array(self.ep_as), self.place)
        tensor_vt = core.LoDTensor()
        tensor_vt.set(discounted_ep_rs_norm, self.place)

        # train on episode
        self.exe.run(
            framework.default_main_program(),
            feed={
                "obs": tensor_obs,  # shape=[None, n_obs]
                "acts": tensor_as,  # shape=[None, ]
                "vt": tensor_vt  # shape=[None, ]
            })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
