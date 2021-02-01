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
                 pad_id,
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
        self.pad_id = pad_id
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
        token_ids, type_ids, pos_ids, generation_mask = inputs[:4]
        src, src_mask = self.gen_input(token_ids, type_ids, pos_ids,
                                       generation_mask)

        if self.is_infer:
            tgt_ids, tgt_pos, tgt_generation_mask = inputs[4:]
            cache = self.encoder.gen_cache(src)
            enc_out, new_cache = self.encoder(src, src_mask, cache)
            pred_ids = self.generate(tgt_ids, tgt_pos, tgt_generation_mask,
                                     new_cache)
            return pred_ids
        else:
            tgt_pos = inputs[4]
            enc_out = self.encoder(src, src_mask)
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

    def calc_logits(self, enc_out, tgt_pos=None):
        # [batch_size * seq_len, d_model]
        enc_out = paddle.reshape(enc_out, [-1, enc_out.shape[-1]])
        # [x, d_model]
        if tgt_pos is not None:
            out = paddle.gather(enc_out, tgt_pos)
        else:
            out = enc_out
        out = self.fc_layer(out)
        out = self.gelu_layer(out)
        out = self.norm_layer(out)
        # [x, vocab_size]
        logits = paddle.matmul(
            out, self.word_embedding_layer.weight,
            transpose_y=True) + self.logits_bias
        return logits

    def generate(self, tgt_ids, tgt_pos, tgt_generation_mask, cache):
        pred_ids = tgt_ids
        cur_len = 0
        unfinished_flag = paddle.full(tgt_ids.shape, 1, 'int64')
        scores = paddle.full(tgt_ids.shape, 0.0, dtype='float32')

        while cur_len < self.max_dec_len:
            append_mask = paddle.cast(
                tgt_ids != self.eos_id, dtype=tgt_generation_mask.dtype)
            tgt_generation_mask = paddle.concat(
                [tgt_generation_mask, paddle.unsqueeze(append_mask, 1)],
                axis=-1)
            tgt_sent = paddle.ones(
                [tgt_generation_mask.shape[0], 1], dtype=tgt_ids.dtype)

            src, src_mask = self.gen_input(tgt_ids, tgt_sent, tgt_pos,
                                           tgt_generation_mask)
            out, cache = self.encoder(src, src_mask, cache)
            # [batch_size, vocab_size]
            logits = self.calc_logits(out)

            # pre-process distribution
            logits[:, self.unk_id] = -1e9
            logits[:, self.mask_id] = -1e9
            logits[:, self.bos_id] = -1e9
            if cur_len < self.min_dec_len:
                logits[:, self.eos_id] = -1e9

            probs = F.softmax(logits)

            # Top-k strategy
            # [batch_size, topk]
            topk_probs, topk_idx = paddle.topk(probs, k=self.topk)
            mask = paddle.cast(probs >= topk_probs[:, -1:], 'float32')
            sums = paddle.sum(topk_probs, axis=-1, keepdim=True)
            new_probs = probs * mask / sums
            # [batch_size, 1]
            next_token_ids = paddle.multinomial(new_probs)
            next_token_score = paddle.index_sample(probs, next_token_ids)
            next_token_score = paddle.log(next_token_score)

            next_token_ids = next_token_ids * unfinished_flag + self.pad_id * (
                1 - unfinished_flag)
            next_token_score = next_token_score * unfinished_flag + scores * (
                1 - unfinished_flag)
            scores = (scores * cur_len + next_token_score) / (cur_len + 1)

            unfinished_flag = unfinished_flag * (next_token_ids != self.eos_id)

            cur_len += 1
            tgt_ids = next_token_ids
            tgt_pos = tgt_pos + 1
            pred_ids = paddle.concat([pred_ids, tgt_ids], axis=1)

            # Stop when there is a </s> in all sentences
            if paddle.max(unfinished_flag) == 0:
                break
        return pred_ids, scores
