"""
Deep Attention Matching Network
"""

import six
import numpy as np
import paddle.fluid as fluid
import layers


class Net(object):
    """
    Deep attention matching network
    """
    def __init__(self, max_turn_num, max_turn_len, vocab_size, emb_size,
                 stack_num, channel1_num, channel2_num):
        """
        Init
        """
        self._max_turn_num = max_turn_num
        self._max_turn_len = max_turn_len
        self._vocab_size = vocab_size
        self._emb_size = emb_size
        self._stack_num = stack_num
        self._channel1_num = channel1_num
        self._channel2_num = channel2_num
        self._feed_names = []
        self.word_emb_name = "shared_word_emb"
        self.use_stack_op = True
        self.use_mask_cache = True
        self.use_sparse_embedding = True

    def create_py_reader(self, capacity, name):
        """
        Create py reader
        """
        # turns ids
        shapes = [[-1, self._max_turn_len, 1]
                  for i in six.moves.xrange(self._max_turn_num)]
        dtypes = ["int64" for i in six.moves.xrange(self._max_turn_num)]
        # turns mask
        shapes += [[-1, self._max_turn_len, 1]
                   for i in six.moves.xrange(self._max_turn_num)]
        dtypes += ["float32" for i in six.moves.xrange(self._max_turn_num)]

        # response ids, response mask, label
        shapes += [[-1, self._max_turn_len, 1], [-1, self._max_turn_len, 1],
                   [-1, 1]]
        dtypes += ["int64", "float32", "float32"]

        py_reader = fluid.layers.py_reader(
            capacity=capacity,
            shapes=shapes,
            lod_levels=[0] * (2 * self._max_turn_num + 3),
            dtypes=dtypes,
            name=name,
            use_double_buffer=True)

        data_vars = fluid.layers.read_file(py_reader)

        self.turns_data = data_vars[0:self._max_turn_num]
        self.turns_mask = data_vars[self._max_turn_num:2 * self._max_turn_num]
        self.response = data_vars[-3]
        self.response_mask = data_vars[-2]
        self.label = data_vars[-1]
        return py_reader

    def create_data_layers(self):
        """
        Create data layer
        """
        self._feed_names = []

        self.turns_data = []
        for i in six.moves.xrange(self._max_turn_num):
            name = "turn_%d" % i
            turn = fluid.layers.data(
                name=name, shape=[self._max_turn_len, 1], dtype="int64")
            self.turns_data.append(turn)
            self._feed_names.append(name)

        self.turns_mask = []
        for i in six.moves.xrange(self._max_turn_num):
            name = "turn_mask_%d" % i
            turn_mask = fluid.layers.data(
                name=name, shape=[self._max_turn_len, 1], dtype="float32")
            self.turns_mask.append(turn_mask)
            self._feed_names.append(name)

        self.response = fluid.layers.data(
            name="response", shape=[self._max_turn_len, 1], dtype="int64")
        self.response_mask = fluid.layers.data(
            name="response_mask",
            shape=[self._max_turn_len, 1],
            dtype="float32")
        self.label = fluid.layers.data(name="label", shape=[1], dtype="float32")
        self._feed_names += ["response", "response_mask", "label"]

    def get_feed_names(self):
        """
        Return feed names
        """
        return self._feed_names

    def set_word_embedding(self, word_emb, place):
        """
        Set word embedding
        """
        word_emb_param = fluid.global_scope().find_var(
            self.word_emb_name).get_tensor()
        word_emb_param.set(word_emb, place)

    def create_network(self):
        """
        Create network
        """
        mask_cache = dict() if self.use_mask_cache else None

        response_emb = fluid.layers.embedding(
            input=self.response,
            size=[self._vocab_size + 1, self._emb_size],
            is_sparse=self.use_sparse_embedding,
            param_attr=fluid.ParamAttr(
                name=self.word_emb_name,
                initializer=fluid.initializer.Normal(scale=0.1)))

        # response part
        Hr = response_emb
        Hr_stack = [Hr]

        for index in six.moves.xrange(self._stack_num):
            Hr = layers.block(
                name="response_self_stack" + str(index),
                query=Hr,
                key=Hr,
                value=Hr,
                d_key=self._emb_size,
                q_mask=self.response_mask,
                k_mask=self.response_mask,
                mask_cache=mask_cache)
            Hr_stack.append(Hr)

        # context part
        sim_turns = []
        for t in six.moves.xrange(self._max_turn_num):
            Hu = fluid.layers.embedding(
                input=self.turns_data[t],
                size=[self._vocab_size + 1, self._emb_size],
                is_sparse=self.use_sparse_embedding,
                param_attr=fluid.ParamAttr(
                    name=self.word_emb_name,
                    initializer=fluid.initializer.Normal(scale=0.1)))
            Hu_stack = [Hu]

            for index in six.moves.xrange(self._stack_num):
                # share parameters
                Hu = layers.block(
                    name="turn_self_stack" + str(index),
                    query=Hu,
                    key=Hu,
                    value=Hu,
                    d_key=self._emb_size,
                    q_mask=self.turns_mask[t],
                    k_mask=self.turns_mask[t],
                    mask_cache=mask_cache)
                Hu_stack.append(Hu)

            # cross attention
            r_a_t_stack = []
            t_a_r_stack = []
            for index in six.moves.xrange(self._stack_num + 1):
                t_a_r = layers.block(
                    name="t_attend_r_" + str(index),
                    query=Hu_stack[index],
                    key=Hr_stack[index],
                    value=Hr_stack[index],
                    d_key=self._emb_size,
                    q_mask=self.turns_mask[t],
                    k_mask=self.response_mask,
                    mask_cache=mask_cache)
                r_a_t = layers.block(
                    name="r_attend_t_" + str(index),
                    query=Hr_stack[index],
                    key=Hu_stack[index],
                    value=Hu_stack[index],
                    d_key=self._emb_size,
                    q_mask=self.response_mask,
                    k_mask=self.turns_mask[t],
                    mask_cache=mask_cache)

                t_a_r_stack.append(t_a_r)
                r_a_t_stack.append(r_a_t)

            t_a_r_stack.extend(Hu_stack)
            r_a_t_stack.extend(Hr_stack)

            if self.use_stack_op:
                t_a_r = fluid.layers.stack(t_a_r_stack, axis=1)
                r_a_t = fluid.layers.stack(r_a_t_stack, axis=1)
            else:
                for index in six.moves.xrange(len(t_a_r_stack)):
                    t_a_r_stack[index] = fluid.layers.unsqueeze(
                        input=t_a_r_stack[index], axes=[1])
                    r_a_t_stack[index] = fluid.layers.unsqueeze(
                        input=r_a_t_stack[index], axes=[1])

                t_a_r = fluid.layers.concat(input=t_a_r_stack, axis=1)
                r_a_t = fluid.layers.concat(input=r_a_t_stack, axis=1)

            # sim shape: [batch_size, 2*(stack_num+1), max_turn_len, max_turn_len]
            sim = fluid.layers.matmul(
                x=t_a_r, y=r_a_t, transpose_y=True, alpha=1 / np.sqrt(200.0))
            sim_turns.append(sim)

        if self.use_stack_op:
            sim = fluid.layers.stack(sim_turns, axis=2)
        else:
            for index in six.moves.xrange(len(sim_turns)):
                sim_turns[index] = fluid.layers.unsqueeze(
                    input=sim_turns[index], axes=[2])
            # sim shape: [batch_size, 2*(stack_num+1), max_turn_num, max_turn_len, max_turn_len]
            sim = fluid.layers.concat(input=sim_turns, axis=2)

        final_info = layers.cnn_3d(sim, self._channel1_num, self._channel2_num)
        loss, logits = layers.loss(final_info, self.label)
        return loss, logits
