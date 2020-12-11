import argparse

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from readers.nsp_reader import NSPReader
from utils.args import parse_args
from tasks.dialog_generation import post_process_context, post_process_response


class Plato2EncoderLayer(nn.Layer):
    def __init__(self, n_head, hidden_size, attn_dropout, act_dropout):
        super(Plato2EncoderLayer, self).__init__()

        self.self_attn = nn.MultiHeadAttention(hidden_size, n_head,
                                               attn_dropout)
        self.pre_norm_layer = nn.LayerNorm(hidden_size)
        self.post_norm_layer = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

        self.dropout_layer = nn.Dropout(act_dropout)
        self.gelu_layer = nn.GELU()

    def forward(self, x, attn_mask, cache):
        query = self.pre_norm_layer(x)
        attn_output, new_cache = self.self_attn(query, None, None, attn_mask,
                                                cache)
        attn_output = self.dropout_layer(attn_output)
        attn_output = attn_output + x
        ffd_input = self.post_norm_layer(attn_output)

        ffd_output = self.fc1(ffd_input)
        ffd_output = self.gelu_layer(ffd_output)
        ffd_output = self.dropout_layer(ffd_output)

        ffd_output = self.fc2(ffd_output)
        ffd_output = self.dropout_layer(ffd_output)
        out = ffd_output + attn_output

        return out, new_cache


class Plato2Encoder(nn.Layer):
    def __init__(self, vocab_size, type_size, max_position_seq_len, num_layers,
                 n_head, hidden_size, attn_dropout, act_dropout):
        super(Plato2Encoder, self).__init__()

        self.n_head = n_head

        self.word_embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.sent_embedding_layer = nn.Embedding(type_size, hidden_size)
        self.pos_embedding_layer = nn.Embedding(max_position_seq_len,
                                                hidden_size)

        self.encoder_layers = []
        for i in range(num_layers):
            encoder_layer = Plato2EncoderLayer(n_head, hidden_size,
                                               attn_dropout, act_dropout)
            self.encoder_layers.append(encoder_layer)
            self.add_sublayer('plato2_encoder_layer_' + str(i))
        self.post_encoder_layer_norm = nn.LayerNorm(hidden_size)

        self.dropout_layer = nn.Dropout(act_dropout)

    def forward(self,
                caches,
                token_ids,
                type_ids,
                pos_ids,
                generation_mask,
                aux_emb=None):
        out, self_attn_mask = self.gen_input(token_ids, type_ids, pos_ids,
                                             generation_mask, aux_emb)

        new_caches = []
        for i, encoder_layer in enumerate(self.encoder_layers):
            out, new_cache = encoder_layers(out, self_attn_mask, caches[i])
            new_caches.append(new_cache)

        enc_output = self.post_encoder_layer_norm(out)
        return enc_output, new_caches

    def gen_input(self, token_ids, type_ids, pos_ids, input_mask, aux_emb=None):
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


class NSP(nn.Layer):
    def __init__(self, ):
        super(NSP, self).__init__()

        self.hidden_size = hidden_size

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, n_head, hidden_size * 4, act_dropout, 'gelu',
            attn_dropout, act_dropout, 'True')
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers,
                                             encoder_norm)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

        self.tanh_layer = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, inputs, label_pos):
        """
        token_ids (20, 108, 1)
        type_ids (20, 108, 1)
        pos_ids (20, 108, 1)
        attention_mask (20, 108, 108)
        label_pos (20, 1)
        data_id (20, 1)
        """
        token_ids = inputs['token_ids']
        type_ids = inputs['type_ids']
        pos_ids = inputs['pos_ids']
        generation_mask = inputs['generation_mask']
        label_pos = inputs["label_pos"]

        out, self_attn_mask = self.gen_input(token_ids, type_ids, pos_ids,
                                             generation_mask)
        # [-1, seq_len, hidden_size]
        enc_out = self.encoder(out, self_attn_mask)

        enc_out = paddle.reshape(enc_out, [-1, self.hidden_size])
        label_pos = paddle.cast(label_pos, 'int64')
        out = paddle.gather(enc_out, label_pos)
        pooled_out = self.fc1(out)
        pooled_out = self.tanh_layer(pooled_out)

        # [-1, 2]
        logits = self.fc2(pooled_out)
        probs = self.softmax(logits)

        return probs

    def gen_input(self, token_ids, type_ids, pos_ids, input_mask, aux_emb=None):
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


class Plato2InferModel(nn.Layer):
    def __init__(self,
                 vocab_size=8001,
                 type_size=2,
                 latent_type_size=20,
                 max_position_seq_len=256,
                 num_layers=24,
                 n_head=16,
                 hidden_size=1024,
                 act_dropout=0.1,
                 attn_dropout=0.1,
                 max_dec_len=64,
                 min_dec_len=1,
                 topk=10):
        super(Plato2InferModel, self).__init__()

        self.num_layers = num_layers
        self.latent_type_size = latent_type_size
        self.max_dec_len = max_dec_len
        self.min_dec_len = min_dec_len
        self.topk = topk
        self.unk_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.mask_id = 8000
        self.after_eos = paddle.ones([vocab_size]) * -1e9
        self.after_eos[self.eos_id] = 0

        self.latent_weight = paddle.create_parameter(
            [hidden_size, latent_type_size], 'float32')

        self.encoder = Plato2Encoder(vocab_size, type_size,
                                     max_position_seq_len, num_layers, n_head,
                                     hidden_size, attn_dropout, act_dropout)

        self.logits_fc_layer = nn.Linear(hidden_size, hidden_size)
        self.logits_layer_norm = nn.LayerNorm(hidden_size)
        self.logits_bias = paddle.create_parameter(
            [vocab_size], 'float32', is_bias=True)

        self.nsp_predictor = NSP()

        self.gelu_layer = nn.GELU()
        self.softmax = nn.Softmax()

        self.data_reader = self.get_reader()

    @paddle.no_grad()
    def forward(self, inputs):
        token_ids = inputs['token_ids']
        type_ids = inputs['type_ids']
        pos_ids = inputs['pos_ids']
        generation_mask = inputs['generation_mask']
        latent_id = inputs['latent_id']
        data_id = inputs['data_id']

        # [-1, 1, latent_type_size]
        latent_id = F.one_hot(latent_id, self.latent_type_size)
        # [-1, 1, hidden_size]
        latent_emb = paddle.matmul(
            latent_id, self.latent_weight, transpose_y=True)

        caches = [
            self.n_multi_att[i].gen_cache(token_ids)
            for i in range(self.num_layers)
        ]
        # [-1, seq_len + 1, hidden_size]
        enc_out, new_caches = self.encoder(caches, token_ids, type_ids, pos_ids,
                                           generation_mask, latent_emb)

        pred_ids = self.decoder(inputs, new_caches)

        data_generator = self.gen_nsp_input(token_ids, pred_ids, data_id)
        for nsp_inputs in data_generator():
            probs = self.nsp_predictor(nsp_inputs)

        return

    def decoder(self, inputs, caches):
        tgt_ids = inputs['tgt_ids']
        tgt_pos = inputs['tgt_pos']
        tgt_generation_mask = inputs['tgt_generation_mask']
        predictions = tgt_ids

        # TODO
        step = 0
        while step < self.max_dec_len:
            # [-1, 1]
            append_mask = paddle.cast(
                tgt_ids != self.eos_id, dtype=tgt_generation_mask.dtype)
            tgt_generation_mask = paddle.concat(
                [tgt_generation_mask, paddle.unsqueeze(append_mask, 1)],
                axis=-1)
            tgt_sent = paddle.ones(
                [tgt_generation_mask.shape[0], 1], dtype=tgt_ids.dtype)

            # [-1, 1, hidden_size]
            out = self.encoder(tgt_ids, tgt_sent, tgt_pos, tgt_generation_mask)
            out = paddle.squeeze(out, axis=1)

            # [-1, hidden_size]
            trans = self.logits_fc_layer(out)
            trans = self.gelu_layer(trans)
            trans = self.logits_layer_norm(trans)

            # [-1, vocab_size]
            logits = paddle.matmul(
                trans,
                self.encoder.word_embedding_layer.weight,
                transpose_y=True) + self.logits_bias
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

    def get_reader(self):
        parser = argparse.ArgumentParser()
        NSPReader.add_cmdline_args(parser)
        args = parse_args(parser)
        args.batch_size *= args.latent_type_size
        args.tokenized_input = True
        reader = NSPReader(args)
        return reader

    def gen_nsp_input(self, token_ids, pred_ids, data_id):
        token_ids = token_ids.numpy()
        pred_ids = pred_ids.numpy()
        data_id = data_id.numpy()
        predictions = []
        for raw, pred, idx in zip(token_ids, pred_ids, data_id):
            info = {
                'response_token_ids': pred,
                'context_token_ids': raw,
                'data_id': idx
            }
            tokens = post_process_context(raw, self.reader)
            pred_token_ids, pred_tokens = post_process_response(
                info["response_token_ids"], self.reader)
            info["context"] = " [SEP] ".join(" ".join(u) for u in tokens)
            info["response"] = " ".join(pred_tokens)
            info["num_token"] = len(pred_token_ids)
            info["cross_turn_repetition"] = get_cross_turn_repetition(
                tokens, pred_tokens, self.reader.eos_id, self.is_cn)
            info["in_turn_repetition"] = max(
                get_in_turn_repetition(pred_tokens, self.is_cn),
                get_in_turn_repetition(pred_token_ids))
            predictions.append(info)

        def __reader__():
            headers = ["src", "tgt", "data_id"]

            Example = namedtuple("Example", headers)

            for i, (raw, pred) in enumerate(zip(token_ids, pred_ids)):
                context = post_process_context(
                    raw, self.data_reader, merge=False)
                _, response = post_process_response(
                    pred, self.data_reader, merge=False)
                context_tokenized_input = " [SEP] ".join(" ".join(utt)
                                                         for utt in context)
                response_tokenized_input = " ".join(response)
                example = Example(
                    src=context_tokenized_input,
                    tgt=response_tokenized_input,
                    data_id=i)
                data = self.data_reader._convert_example_to_record(
                    example, is_infer=True)
                yield data
            return

        generator = self.data_reader.data_generator(
            reader=__reader__,
            is_infer=True,
            phase="test", )
        return generator


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
        #print(key, inputs[key].shape, inputs[key].dtype)

    model = InferPlato2()

    ckpt_path = './data/pretrain.pdparams'
    state_dict = paddle.load(ckpt_path)
    model.set_state_dict(state_dict)

    model.eval()

    dec_out = model(inputs)
