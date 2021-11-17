import paddle.fluid as fluid


class Model:
    def __init__(self,
                 user_slots_num,
                 pos_doc_slots_num,
                 neg_doc_slots_num,
                 dict_size_,
                 emb_size_,
                 random_ratio_=1,
                 is_distributed_=False):
        self._dict_dim = dict_size_
        self._emb_size = emb_size_
        self._is_distributed = is_distributed_
        self.random_ratio = random_ratio_

        self._all_slots = []
        self._train_program = fluid.Program()
        self._startup_program = fluid.Program()
        with fluid.program_guard(self._train_program, self._startup_program):
            with fluid.unique_name.guard():
                self.matchnet(user_slots_num, pos_doc_slots_num,
                              neg_doc_slots_num)

    def matchnet(self, user_slots_num, positive_doc_slots_num,
                 negative_doc_slots_num):
        def embedding_layer(input):
            emb = fluid.layers.embedding(
                input=input,
                is_sparse=True,
                is_distributed=self._is_distributed,
                size=[self._dict_dim, self._emb_size + 1],
                param_attr=fluid.ParamAttr(
                    name="embedding", initializer=fluid.initializer.Uniform()))
            bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
            return bow

        # user features
        user_embs = []
        for i in range(user_slots_num):
            slot = fluid.layers.data(
                name="user_slot_" + str(i + 1),
                shape=[1],
                dtype="int64",
                lod_level=1)
            self._all_slots.append(slot)
            slot_emb = embedding_layer(slot)
            user_embs.append(slot_emb)
        self._u_sum = fluid.layers.sums(input=user_embs)
        self.user_emb = fluid.layers.slice(
            self._u_sum, axes=[1], starts=[1], ends=[self._emb_size + 1])

        # positive doc features
        n1_doc_embs = []
        for i in range(positive_doc_slots_num):
            slot = fluid.layers.data(
                name="pos_doc_slot_" + str(i),
                shape=[1],
                dtype="int64",
                lod_level=1)
            self._all_slots.append(slot)
            slot_emb = embedding_layer(slot)
            n1_doc_embs.append(slot_emb)
        self._n1_sum = fluid.layers.sums(input=n1_doc_embs)
        self.n1_emb_bias = fluid.layers.slice(
            self._n1_sum, axes=[1], starts=[0], ends=[1])
        self.n1_emb = fluid.layers.slice(
            self._n1_sum, axes=[1], starts=[1], ends=[self._emb_size + 1])

        # netative doc features
        n2_doc_embs = []
        for i in range(negative_doc_slots_num):
            slot = fluid.layers.data(
                name="neg_doc_slot_" + str(i),
                shape=[1],
                dtype="int64",
                lod_level=1)
            self._all_slots.append(slot)
            slot_emb = embedding_layer(slot)
            n2_doc_embs.append(slot_emb)
        self._n2_sum = fluid.layers.sums(input=n2_doc_embs)
        self.n2_emb_bias = fluid.layers.slice(
            self._n2_sum, axes=[1], starts=[0], ends=[1])
        self.n2_emb = fluid.layers.slice(
            self._n2_sum, axes=[1], starts=[1], ends=[self._emb_size + 1])

        # similariry between user and positive doc
        _cross_p = fluid.layers.reduce_sum(
            fluid.layers.elementwise_mul(self.user_emb, self.n1_emb),
            dim=1,
            keep_dim=True)
        self.similarity_p = fluid.layers.sums([_cross_p, self.n1_emb_bias])

        # similariry between user and negative doc
        _cross_n = fluid.layers.reduce_sum(
            fluid.layers.elementwise_mul(self.user_emb, self.n2_emb),
            dim=1,
            keep_dim=True)
        self.similarity_n = fluid.layers.sums([_cross_n, self.n2_emb_bias])

        # scale
        self.similarity_p_scale = fluid.layers.concat(
            [self.similarity_p] * self.random_ratio, axis=0)
        simialrity_n_scale_list = [self.similarity_n]
        for _ in range(self.random_ratio - 1):
            _n2_emb_shuffle = fluid.contrib.layers.shuffle_batch(self.n2_emb)
            _n2_emb_bias_shuffle = fluid.contrib.layers.shuffle_batch(
                self.n2_emb_bias)
            _cross_n_shuffle = fluid.layers.reduce_sum(
                fluid.layers.elementwise_mul(self.user_emb, _n2_emb_shuffle),
                dim=1,
                keep_dim=True)
            simialrity_n_scale_list.append(
                fluid.layers.sums([_cross_n_shuffle, _n2_emb_bias_shuffle]))
        self.similarity_n_scale = fluid.layers.concat(
            simialrity_n_scale_list, axis=0)

        # loss function
        self._sim = fluid.layers.elementwise_sub(self.similarity_p_scale,
                                                 self.similarity_n_scale)
        random_rematch_label_target = fluid.layers.fill_constant_batch_size_like(
            input=self._sim, shape=[-1, 1], value=1.0, dtype='float32')
        logits_sim = fluid.layers.sigmoid(
            fluid.layers.clip(
                self._sim, min=-15.0, max=15.0),
            name="similarity_norm")
        loss = fluid.layers.log_loss(logits_sim, random_rematch_label_target)
        self.avg_cost = fluid.layers.mean(self.random_ratio * loss)
        self.pn, self.correct, self.wrong = self.get_global_pn(
            self.similarity_p, self.similarity_n)

    def get_global_pn(self, pos_score, neg_score):
        wrong = fluid.layers.cast(
            fluid.layers.less_equal(pos_score, neg_score), dtype='float32')
        wrong_cnt = fluid.layers.reduce_sum(wrong)
        right = fluid.layers.cast(
            fluid.layers.less_than(neg_score, pos_score), dtype='float32')
        right_cnt = fluid.layers.reduce_sum(right)

        pn = fluid.layers.elementwise_div(
            right_cnt + 1, wrong_cnt + 1, name="nearline_pn")
        return pn, right_cnt, wrong_cnt
