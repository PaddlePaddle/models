import paddle.fluid as fluid
import paddle.fluid.layers as layers
from model import PolicyGradient

class SeqPGAgent(object):
    def __init__(self,
                 model_cls,
                 reward_func,
                 alg_cls=PolicyGradient,
                 model_hparams={},
                 alg_hparams={},
                 executor=None):
        self.build_program(model_cls, reward_func, alg_cls, model_hparams,
                           alg_hparams)
        self.executor = executor

    def build_program(self, model_cls, reward_func, alg_cls, model_hparams,
                      alg_hparams):
        self.full_program = fluid.Program()
        with fluid.program_guard(self.full_program,
                                 fluid.default_startup_program()):
            source = fluid.data(name="src", shape=[None, None], dtype="int64")
            source_length = fluid.data(name="src_sequence_length",
                                       shape=[None],
                                       dtype="int64")
            self.alg = alg_cls(model=model_cls(**model_hparams), **alg_hparams)
            self.probs, self.samples, self.sample_length = self.alg.predict(
                source, source_length)
            self.samples.stop_gradient = True
            reward = fluid.layers.create_global_var(
                name="reward",
                shape=[-1, -1],  # batch_size, seq_len
                value="0",
                dtype=self.probs.dtype)
            self.reward = fluid.layers.py_func(
                func=reward_func,
                x=[self.samples, self.sample_length],
                out=reward)
            self.cost = self.alg.learn(self.probs, self.samples, self.reward,
                                       self.sample_length)

        # to define the same parameters between different programs
        self.pred_program = self.full_program._prune_with_input(
            [source.name, source_length.name],
            [self.probs, self.samples, self.sample_length])

    def predict(self, feed_dict):
        samples, sample_length = self.executor.run(
            self.pred_program,
            feed=feed_dict,
            fetch_list=[self.samples, self.sample_length])
        return samples, sample_length

    def learn(self, feed_dict):
        reward, cost = self.executor.run(self.full_program,
                                         feed=feed_dict,
                                         fetch_list=[self.reward, self.cost])
        return reward, cost
