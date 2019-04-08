import numpy as np
import paddle.fluid as fluid
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

        self.place = fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)

    def build_net(self):

        obs = fluid.layers.data(
            name='obs', shape=[self.n_features], dtype='float32')
        acts = fluid.layers.data(name='acts', shape=[1], dtype='int64')
        vt = fluid.layers.data(name='vt', shape=[1], dtype='float32')
        # fc1
        fc1 = fluid.layers.fc(input=obs, size=10, act="tanh")  # tanh activation
        # fc2
        all_act_prob = fluid.layers.fc(input=fc1,
                                       size=self.n_actions,
                                       act="softmax")
        self.inferece_program = fluid.defaul_main_program().clone()
        # to maximize total reward (log_p * R) is to minimize -(log_p * R)
        neg_log_prob = fluid.layers.cross_entropy(
            input=self.all_act_prob,
            label=acts)  # this is negative log of chosen action
        neg_log_prob_weight = fluid.layers.elementwise_mul(x=neg_log_prob, y=vt)
        loss = fluid.layers.reduce_mean(
            neg_log_prob_weight)  # reward guided loss

        sgd_optimizer = fluid.optimizer.SGD(self.lr)
        sgd_optimizer.minimize(loss)
        self.exe.run(fluid.default_startup_program())

    def choose_action(self, observation):
        prob_weights = self.exe.run(self.inferece_program,
                                    feed={"obs": observation[np.newaxis, :]},
                                    fetch_list=[self.all_act_prob])
        prob_weights = np.array(prob_weights[0])
        # select action w.r.t the actions prob
        action = np.random.choice(
            range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        tensor_obs = np.vstack(self.ep_obs).astype("float32")
        tensor_as = np.array(self.ep_as).astype("int64")
        tensor_as = tensor_as.reshape([tensor_as.shape[0], 1])
        tensor_vt = discounted_ep_rs_norm.astype("float32")[:, np.newaxis]
        # train on episode
        self.exe.run(
            fluid.default_main_program(),
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
