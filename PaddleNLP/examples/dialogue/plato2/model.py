import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Plato2(nn.Layer):
    def __init__(self,
                 vocab_size=8001,
                 type_size=2,
                 latent_type_size=20,
                 max_position_seq_len=256,
                 n_layer=24,
                 n_head=16,
                 hidden_size=1024,
                 hidden_dropout_prob=0.1,
                 attention_dropout=0.1,
                 max_dec_len=64,
                 min_dec_len=1,
                 topk=10):
        super(Plato2, self).__init__()

        self.n_layer = n_layer
        self.n_head = n_head
        self.latent_type_size = latent_type_size
        self.max_dec_len = max_dec_len
        self.min_dec_len = min_dec_len
        self.topk = topk

        self.mask_id = 8000
        self.bos_id = 1
        self.eos_id = 2

        self.latent_weight = paddle.create_parameter(
            [hidden_size, latent_type_size], 'float32')
        self.word_embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.sent_embedding_layer = nn.Embedding(type_size, hidden_size)
        self.pos_embedding_layer = nn.Embedding(max_position_seq_len,
                                                hidden_size)
        self.dropout_layer = nn.Dropout(hidden_dropout_prob)
        self.post_encoder_layer_norm = nn.LayerNorm(hidden_size)
        self.gelu_layer = nn.GELU()

        self.n_pre_norm_layers = []
        self.n_multi_att = []
        self.n_post_norm_layers = []
        self.n_linear0_layers = []
        self.n_linear1_layers = []
        for i in range(n_layer):
            pre_norm_layer = nn.LayerNorm(hidden_size)
            self.n_pre_norm_layers.append(pre_norm_layer)
            self.add_sublayer('encoder_layer_' + str(i) + '_pre_att_layer_norm',
                              pre_norm_layer)

            multi_att = nn.MultiHeadAttention(hidden_size, n_head,
                                              attention_dropout)
            self.n_multi_att.append(multi_att)
            self.add_sublayer('encoder_layer_' + str(i) + '_multi_head_att',
                              multi_att)

            post_norm_layer = nn.LayerNorm(hidden_size)
            self.n_post_norm_layers.append(post_norm_layer)
            self.add_sublayer('encoder_layer_' + str(i) + '_pre_ffn_layer_norm',
                              post_norm_layer)

            linear0_layer = nn.Linear(hidden_size, hidden_size * 4)
            self.n_linear0_layers.append(linear0_layer)
            self.add_sublayer('encoder_layer_' + str(i) + '_ffn_fc_0',
                              linear0_layer)

            linear1_layer = nn.Linear(hidden_size * 4, hidden_size)
            self.n_linear1_layers.append(linear1_layer)
            self.add_sublayer('encoder_layer_' + str(i) + '_ffn_fc_1',
                              linear1_layer)

        self.logits_fc_layer = nn.Linear(hidden_size, hidden_size)
        self.logits_layer_norm = nn.LayerNorm(hidden_size)
        self.logits_bias = paddle.create_parameter(
            [vocab_size], 'float32', is_bias=True)

        self.token_penalty = paddle.zeros([vocab_size])
        self.token_penalty[self.mask_id] = -1e9
        self.eos_penalty = paddle.zeros([vocab_size])
        self.eos_penalty[self.eos_id] = -1e9

        self.softmax = nn.Softmax()

    def forward(self, inputs):
        token_ids = inputs['token_ids']
        type_ids = inputs['type_ids']
        pos_ids = inputs['pos_ids']
        generation_mask = inputs['generation_mask']
        latent_id = inputs['latent_id']

        # [-1, 1, latent_type_size]
        latent_id = F.one_hot(latent_id, self.latent_type_size)
        # [-1, 1, hidden_size]
        latent_emb = paddle.matmul(
            latent_id, self.latent_weight, transpose_y=True)

        self.caches = [
            self.n_multi_att[i].gen_cache(token_ids)
            for i in range(self.n_layer)
        ]
        # [-1, seq_len + 1, hidden_size]
        enc_out = self.encoder(token_ids, type_ids, pos_ids, generation_mask,
                               latent_emb)

        dec_out = self.decoder(inputs)

        return dec_out

    def encoder(self,
                token_ids,
                type_ids,
                pos_ids,
                generation_mask,
                aux_emb=None):
        # [-1, seq_len + 1, hidden_size]  [-1, n_head, seq_len + 1, seq_len + 1]
        out, n_head_self_attn_mask = self._gen_input(
            token_ids, type_ids, pos_ids, generation_mask, aux_emb)
        for i in range(self.n_layer):
            # [-1, seq_len + 1, hidden_size]
            query = self.n_pre_norm_layers[i](out)
            attn_output, self.caches[i] = self.n_multi_att[i](
                query,
                query,
                query,
                n_head_self_attn_mask,
                cache=self.caches[i])
            attn_output = self.dropout_layer(attn_output)
            attn_output = attn_output + out

            ffd_input = self.n_post_norm_layers[i](attn_output)
            ffd_output = self.n_linear0_layers[i](ffd_input)
            ffd_output = self.gelu_layer(ffd_output)
            ffd_output = self.dropout_layer(ffd_output)
            ffd_output = self.n_linear1_layers[i](ffd_output)
            ffd_output = self.dropout_layer(ffd_output)
            out = ffd_output + attn_output
        enc_output = self.post_encoder_layer_norm(out)
        return enc_output

    def _gen_input(self, token_ids, type_ids, pos_ids, input_mask,
                   aux_emb=None):
        token_emb_out = self.word_embedding_layer(token_ids)
        type_emb_out = self.sent_embedding_layer(type_ids)
        pos_emb_out = self.pos_embedding_layer(pos_ids)
        emb_out = token_emb_out + type_emb_out + pos_emb_out

        # auxiliary memory embeddings
        if aux_emb is not None:
            emb_out = paddle.concat([aux_emb, emb_out], axis=1)

        emb_out = self.dropout_layer(emb_out)

        # generate n-head self-attention mask
        self_attn_mask = input_mask
        self_attn_mask = paddle.scale(
            x=self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = paddle.stack(
            x=[self_attn_mask] * self.n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        return emb_out, n_head_self_attn_mask

    def decoder(self, inputs):
        tgt_ids = inputs['tgt_ids']
        tgt_pos = inputs['tgt_pos']
        tgt_generation_mask = inputs['tgt_generation_mask']
        scores = inputs['init_score']

        # TODO
        step = 0
        while step < self.max_dec_len:
            append_mask = paddle.ones(
                [batch_size, 1, 1], dtype=tgt_generation_mask.dtype)
            tgt_generation_mask = paddle.concat(
                [tgt_generation_mask, append_mask], axis=-1)
            tgt_sent = paddle.ones(
                [tgt_generation_mask.shape[0], 1], dtype=tgt_ids.dtype)
            tgt_pos = paddle.ones(
                [tgt_generation_mask.shape[0], 1],
                dtype=tgt_ids.dtype) * step + tgt_pos

            # [-1, 1, hidden_size]
            out = self.encoder(tgt_ids, tgt_sent, tgt_pos, tgt_generation_mask)
            out = paddle.squeeze(out, axis=1)

            # [-1, hidden_size]
            trans = self.logits_fc_layer(out)
            trans = self.gelu_layer(out)
            trans = self.logits_layer_norm(out)

            # [-1, vocab_size]
            logits = paddle.matmul(
                trans, self.word_embedding_layer.weight,
                transpose_y=True) + self.logits_bias
            logits += self.token_penalty
            if step < self.min_dec_len:
                logits += self.eos_penalty

            probs = self.softmax(logits)

            # [-1, topk]
            topk_probs, _ = paddle.topk(probs, k=self.topk)
            new_probs = probs * paddle.cast(probs >= topk_probs[:, -1],
                                            'float32') / paddle.sum(
                                                topk_probs,
                                                axis=-1,
                                                keepdim=True)
            sampling_ids = paddle.multinomial(new_probs)

            pre_len = step
            cur_len = step = step + 1

            accu_scores = (paddle.log(topk_scores) + scores * pre_len) / cur_len

            # TODO beam_search & gather


if __name__ == '__main__':
    import pickle
    import numpy as np
    with open('./data/inputs.pickle', 'rb') as fin:
        inputs = pickle.load(fin)
    """
    token_ids (200, 120, 1)
    type_ids (200, 120, 1)
    pos_ids (200, 120, 1)
    generation_mask (200, 121, 121)
    init_score -> List (200, 1)
    tgt_ids -> List (200, 1, 1)
    tgt_pos -> List (200, 1, 1)
    parent_idx (200,)
    tgt_generation_mask (200, 1, 121)
    data_id (200, 1)
    latent_id (200, 1)
    """
    for key in inputs:
        inputs[key] = paddle.to_tensor(inputs[key])
        if key in ['token_ids', 'type_ids', 'pos_ids', 'tgt_ids', 'tgt_pos']:
            inputs[key] = paddle.squeeze(inputs[key], axis=-1)
        print(key, inputs[key].shape, inputs[key].dtype)

    model = Plato2()
    model(inputs)
