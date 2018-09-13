import cPickle as pickle
import numpy as np
import paddle.fluid as fluid
import utils.layers as layers


class Net(object):
    def __init__(self, max_turn_num, max_turn_len, vocab_size, emb_size,
                 stack_num):

        self._max_turn_num = max_turn_num
        self._max_turn_len = max_turn_len
        self._vocab_size = vocab_size
        self._emb_size = emb_size
        self._stack_num = stack_num
        self.word_emb_name = "shared_word_emb"
        self.use_stack_op = True
        self.use_mask_cache = True
        self.use_sparse_embedding = True

    def create_network(self):
        mask_cache = dict() if self.use_mask_cache else None

        turns_data = []
        for i in xrange(self._max_turn_num):
            turn = fluid.layers.data(
                name="turn_%d" % i,
                shape=[self._max_turn_len, 1],
                dtype="int32")
            turns_data.append(turn)

        turns_mask = []
        for i in xrange(self._max_turn_num):
            turn_mask = fluid.layers.data(
                name="turn_mask_%d" % i,
                shape=[self._max_turn_len, 1],
                dtype="float32")
            turns_mask.append(turn_mask)

        response = fluid.layers.data(
            name="response", shape=[self._max_turn_len, 1], dtype="int32")
        response_mask = fluid.layers.data(
            name="response_mask",
            shape=[self._max_turn_len, 1],
            dtype="float32")
        label = fluid.layers.data(name="label", shape=[1], dtype="float32")

        response_emb = fluid.layers.embedding(
            input=response,
            size=[self._vocab_size + 1, self._emb_size],
            is_sparse=self.use_sparse_embedding,
            param_attr=fluid.ParamAttr(
                name=self.word_emb_name,
                initializer=fluid.initializer.Normal(scale=0.1)))

        # response part
        Hr = response_emb
        Hr_stack = [Hr]

        for index in range(self._stack_num):
            Hr = layers.block(
                name="response_self_stack" + str(index),
                query=Hr,
                key=Hr,
                value=Hr,
                d_key=self._emb_size,
                q_mask=response_mask,
                k_mask=response_mask,
                mask_cache=mask_cache)
            Hr_stack.append(Hr)

        # context part
        sim_turns = []
        for t in xrange(self._max_turn_num):
            Hu = fluid.layers.embedding(
                input=turns_data[t],
                size=[self._vocab_size + 1, self._emb_size],
                is_sparse=self.use_sparse_embedding,
                param_attr=fluid.ParamAttr(
                    name=self.word_emb_name,
                    initializer=fluid.initializer.Normal(scale=0.1)))
            Hu_stack = [Hu]

            for index in range(self._stack_num):
                # share parameters
                Hu = layers.block(
                    name="turn_self_stack" + str(index),
                    query=Hu,
                    key=Hu,
                    value=Hu,
                    d_key=self._emb_size,
                    q_mask=turns_mask[t],
                    k_mask=turns_mask[t],
                    mask_cache=mask_cache)
                Hu_stack.append(Hu)

            # cross attention 
            r_a_t_stack = []
            t_a_r_stack = []
            for index in range(self._stack_num + 1):
                t_a_r = layers.block(
                    name="t_attend_r_" + str(index),
                    query=Hu_stack[index],
                    key=Hr_stack[index],
                    value=Hr_stack[index],
                    d_key=self._emb_size,
                    q_mask=turns_mask[t],
                    k_mask=response_mask,
                    mask_cache=mask_cache)
                r_a_t = layers.block(
                    name="r_attend_t_" + str(index),
                    query=Hr_stack[index],
                    key=Hu_stack[index],
                    value=Hu_stack[index],
                    d_key=self._emb_size,
                    q_mask=response_mask,
                    k_mask=turns_mask[t],
                    mask_cache=mask_cache)

                t_a_r_stack.append(t_a_r)
                r_a_t_stack.append(r_a_t)

            t_a_r_stack.extend(Hu_stack)
            r_a_t_stack.extend(Hr_stack)

            if self.use_stack_op:
                t_a_r = fluid.layers.stack(t_a_r_stack, axis=1)
                r_a_t = fluid.layers.stack(r_a_t_stack, axis=1)
            else:
                for index in xrange(len(t_a_r_stack)):
                    t_a_r_stack[index] = fluid.layers.unsqueeze(
                        input=t_a_r_stack[index], axes=[1])
                    r_a_t_stack[index] = fluid.layers.unsqueeze(
                        input=r_a_t_stack[index], axes=[1])

                t_a_r = fluid.layers.concat(input=t_a_r_stack, axis=1)
                r_a_t = fluid.layers.concat(input=r_a_t_stack, axis=1)

            # sim shape: [batch_size, 2*(stack_num+2), max_turn_len, max_turn_len]    
            sim = fluid.layers.matmul(x=t_a_r, y=r_a_t, transpose_y=True)
            sim = fluid.layers.scale(x=sim, scale=1 / np.sqrt(200.0))
            sim_turns.append(sim)

        if self.use_stack_op:
            sim = fluid.layers.stack(sim_turns, axis=2)
        else:
            for index in xrange(len(sim_turns)):
                sim_turns[index] = fluid.layers.unsqueeze(
                    input=sim_turns[index], axes=[2])
            # sim shape: [batch_size, 2*(stack_num+2), max_turn_num, max_turn_len, max_turn_len]
            sim = fluid.layers.concat(input=sim_turns, axis=2)

        # for douban
        final_info = layers.cnn_3d(sim, 32, 16)
        loss, logits = layers.loss(final_info, label)
        return loss, logits
