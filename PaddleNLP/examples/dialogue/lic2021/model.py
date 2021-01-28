from collections import namedtuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BaselineModel(nn.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dropout,
                 activation,
                 normalize_before,
                 vocab_size,
                 type_size,
                 max_seq_len,
                 min_dec_len,
                 max_dec_len,
                 topk,
                 unk_id,
                 bos_id,
                 eos_id,
                 mask_id,
                 is_infer=False):
        super(BaselineModel, self).__init__()

        self.nhead = nhead
        self.min_dec_len = min_dec_len
        self.max_dec_len = max_dec_len
        self.topk = topk
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.mask_id = mask_id
        self.after_eos = paddle.ones([vocab_size]) * -1e9
        self.after_eos[eos_id] = 0
        self.is_infer = is_infer

        self.word_embedding_layer = nn.Embedding(vocab_size, d_model)
        self.sent_embedding_layer = nn.Embedding(type_size, d_model)
        self.pos_embedding_layer = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            d_model * 4,
            dropout,
            activation,
            normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers,
                                             encoder_norm)

        self.fc_layer = nn.Linear(d_model, d_model)
        self.norm_layer = nn.LayerNorm(d_model)
        self.logits_bias = paddle.create_parameter(
            [vocab_size], 'float32', is_bias=True)

        self.dropout_layer = nn.Dropout(dropout)
        self.gelu_layer = nn.GELU()
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        token_ids, type_ids, pos_ids, generation_mask, tgt_pos = inputs

        src, src_mask = self.gen_input(token_ids, type_ids, pos_ids,
                                       generation_mask)

        cache = None
        if self.is_infer:
            cache = self.encoder.gen_cache(src)

        if self.is_infer:
            enc_out, new_cache = self.encoder(src, src_mask, cache)
        else:
            enc_out = self.encoder(src, src_mask)

        if self.is_infer:
            pred_ids = self.generate(inputs, new_cache)
            return pred_ids
        else:
            logits = self.calc_logits(enc_out, tgt_pos)
            return logits

    def gen_input(self, token_ids, type_ids, pos_ids, input_mask):
        token_emb_out = self.word_embedding_layer(token_ids)
        type_emb_out = self.sent_embedding_layer(type_ids)
        pos_emb_out = self.pos_embedding_layer(pos_ids)

        emb_out = token_emb_out + type_emb_out + pos_emb_out
        emb_out = self.dropout_layer(emb_out)

        # generate n-head self-attention mask
        self_attn_mask = input_mask
        self_attn_mask = paddle.scale(
            x=self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = paddle.stack(
            x=[self_attn_mask] * self.nhead, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        return emb_out, n_head_self_attn_mask

    def calc_logits(self, enc_out, tgt_pos):
        # [batch_size * seq_len, d_model]
        enc_out = paddle.reshape(enc_out, [-1, enc_out.shape[-1]])
        # [x, d_model]
        out = paddle.gather(enc_out, tgt_pos)
        out = self.fc_layer(out)
        out = self.gelu_layer(out)
        out = self.norm_layer(out)
        # [x, vocab_size]
        logits = paddle.matmul(
            out, self.word_embedding_layer.weight,
            transpose_y=True) + self.logits_bias
        return logits

    def generate(self, inputs, cache):
        tgt_ids = inputs['tgt_ids']
        tgt_pos = inputs['tgt_pos']
        tgt_generation_mask = inputs['tgt_generation_mask']
        predictions = tgt_ids

        step = 0
        while step < self.max_dec_len:
            append_mask = paddle.cast(
                tgt_ids != self.eos_id, dtype=tgt_generation_mask.dtype)
            tgt_generation_mask = paddle.concat(
                [tgt_generation_mask, paddle.unsqueeze(append_mask, 1)],
                axis=-1)
            tgt_sent = paddle.ones(
                [tgt_generation_mask.shape[0], 1], dtype=tgt_ids.dtype)

            out, cache = self.encoder(cache, tgt_ids, tgt_sent, tgt_pos,
                                      tgt_generation_mask)
            logits = self.calc_logits(out)

            logits[:, self.unk_id] = -1e9
            logits[:, self.bos_id] = -1e9
            logits[:, self.mask_id] = -1e9
            if step < self.min_dec_len:
                logits[:, self.eos_id] = -1e9
            logits = logits * append_mask + (1 - append_mask) * self.after_eos
            probs = self.softmax(logits)

            # [-1, topk]
            topk_probs, _ = paddle.topk(probs, k=self.topk)
            mask = paddle.cast(probs >= topk_probs[:, -1:], 'float32')
            sums = paddle.sum(topk_probs, axis=-1, keepdim=True)
            new_probs = probs * mask / sums
            # [-1, 1]
            sampling_ids = paddle.multinomial(new_probs)

            step = step + 1
            tgt_ids = sampling_ids
            tgt_pos = tgt_pos + 1
            predictions = paddle.concat([predictions, tgt_ids], axis=1)
        return predictions
