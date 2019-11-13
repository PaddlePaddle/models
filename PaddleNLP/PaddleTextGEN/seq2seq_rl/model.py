import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers


class EncoderCell(layers.RNNCell):
    def __init__(self, num_layers, hidden_size, dropout_prob=0.):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = [
            layers.LSTMCell(hidden_size) for i in range(num_layers)
        ]

    def call(self, step_input, states):
        new_states = []
        for i in range(self.num_layers):
            out, new_state = self.lstm_cells[i](step_input, states[i])
            step_input = layers.dropout(
                out, self.dropout_prob) if self.dropout_prob > 0 else out
            new_states.append(new_state)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


class DecoderCell(layers.RNNCell):
    def __init__(self, num_layers, hidden_size, dropout_prob=0.):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = [
            layers.LSTMCell(hidden_size) for i in range(num_layers)
        ]

    def attention(self, hidden, encoder_output, encoder_padding_mask):
        query = layers.fc(hidden,
                          size=encoder_output.shape[-1],
                          bias_attr=False)
        attn_scores = layers.matmul(layers.unsqueeze(query, [1]),
                                    encoder_output,
                                    transpose_y=True)
        if encoder_padding_mask is not None:
            attn_scores = layers.elementwise_add(attn_scores,
                                                 encoder_padding_mask)
        attn_scores = layers.softmax(attn_scores)
        attn_out = layers.squeeze(layers.matmul(attn_scores, encoder_output),
                                  [1])
        attn_out = layers.concat([attn_out, hidden], 1)
        attn_out = layers.fc(attn_out, size=self.hidden_size, bias_attr=False)
        return attn_out

    def call(self,
             step_input,
             states,
             encoder_output,
             encoder_padding_mask=None):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = layers.concat([step_input, input_feed], 1)
        for i in range(self.num_layers):
            out, new_lstm_state = self.lstm_cells[i](step_input, lstm_states[i])
            step_input = layers.dropout(
                out, self.dropout_prob) if self.dropout_prob > 0 else out
            new_lstm_states.append(new_lstm_state)
        out = self.attention(step_input, encoder_output, encoder_padding_mask)
        return out, [new_lstm_states, out]


class Encoder(object):
    def __init__(self,num_layers, hidden_size, dropout_prob=0.):
        self.encoder_cell = EncoderCell(num_layers, hidden_size, dropout_prob)

    def __call__(self, src_emb, src_sequence_length):
        encoder_output, encoder_final_state = layers.rnn(
            cell=self.encoder_cell,
            inputs=src_emb,
            sequence_length=src_sequence_length,
            is_reverse=False)
        return encoder_output, encoder_final_state


class Decoder(object):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 dropout_prob,
                 decoding_strategy="infer_sample",
                 max_decoding_length=100):
        self.decoder_cell = DecoderCell(num_layers, hidden_size, dropout_prob)
        self.decoding_strategy = decoding_strategy
        self.max_decoding_length = None if (
            self.decoding_strategy == "train_greedy") else max_decoding_length

    def __call__(self, decoder_initial_states, encoder_output,
                 encoder_padding_mask, **kwargs):
        output_layer = kwargs.pop("output_layer", None)
        if self.decoding_strategy == "train_greedy":
            # for teach-forcing MLE pre-training
            helper = layers.TrainingHelper(**kwargs)
        elif self.decoding_strategy == "infer_sample":
            helper = layers.SampleEmbeddingHelper(**kwargs)
        elif self.decoding_strategy == "infer_greedy":
            helper = layers.GreedyEmbeddingHelper(**kwargs)
        else:
            # TODO: Add beam_search training support.
            raise ValueError("Unknown decoding strategy: {}".format(
                self.decoding_strategy))
        decoder = layers.BasicDecoder(self.decoder_cell,
                                      helper,
                                      output_fn=output_layer)
        (decoder_output, decoder_final_state,
         dec_seq_lengths) = layers.dynamic_decode(
             decoder,
             inits=decoder_initial_states,
             max_step_num=self.max_decoding_length,
             encoder_output=encoder_output,
             encoder_padding_mask=encoder_padding_mask)
        return decoder_output, decoder_final_state, dec_seq_lengths


class Seq2SeqModel(object):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 dropout_prob,
                 src_vocab_size,
                 trg_vocab_size,
                 bos_token,
                 eos_token,
                 decoding_strategy="infer_sample",
                 max_decoding_length=100):
        self.bos_token, self.eos_token = bos_token, eos_token
        self.src_embeder = lambda x: fluid.embedding(
            input=x,
            size=[src_vocab_size, hidden_size],
            dtype="float32",
            param_attr=fluid.ParamAttr(name="source_embedding"))
        self.trg_embeder = lambda x: fluid.embedding(
            input=x,
            size=[trg_vocab_size, hidden_size],
            dtype="float32",
            param_attr=fluid.ParamAttr(name="target_embedding"))
        self.encoder = Encoder(num_layers, hidden_size, dropout_prob)
        self.decoder = Decoder(num_layers, hidden_size, dropout_prob,
                               decoding_strategy, max_decoding_length)
        self.output_layer = lambda x: layers.fc(
            x,
            size=trg_vocab_size,
            num_flatten_dims=len(x.shape) - 1,
            param_attr=fluid.ParamAttr(name="output_w"),
            bias_attr=False)

    def __call__(self, src, src_length, trg=None, trg_length=None):
        # encoder
        encoder_output, encoder_final_state = self.encoder(
            self.src_embeder(src), src_length)

        decoder_initial_states = [
            encoder_final_state,
            self.decoder.decoder_cell.get_initial_states(
                batch_ref=encoder_output, shape=[encoder_output.shape[-1]])
        ]
        src_mask = layers.sequence_mask(src_length,
                                        maxlen=layers.shape(src)[1],
                                        dtype="float32")
        encoder_padding_mask = (src_mask - 1.0) * 1e9
        encoder_padding_mask = layers.unsqueeze(encoder_padding_mask, [1])

        # decoder
        decoder_kwargs = {
            "inputs": self.trg_embeder(trg),
            "sequence_length": trg_length,
        } if self.decoder.decoding_strategy == "train_greedy" else {
            "embedding_fn":
            self.trg_embeder,
            "start_tokens":
            layers.fill_constant_batch_size_like(input=encoder_output,
                                                 shape=[-1],
                                                 dtype=src.dtype,
                                                 value=self.bos_token),
            "end_token":
            self.eos_token
        }
        decoder_kwargs["output_layer"] = self.output_layer

        (decoder_output, decoder_final_state,
         dec_seq_lengths) = self.decoder(decoder_initial_states, encoder_output,
                                         encoder_padding_mask, **decoder_kwargs)
        logits, samples, sample_length = (decoder_output.cell_outputs,
                                          decoder_output.sample_ids,
                                          dec_seq_lengths)
        probs = layers.softmax(logits)
        return probs, samples, sample_length


class PolicyGradient(object):
    def __init__(self, model, lr=None):
        self.model = model
        self.lr = lr

    def predict(self, src, src_length):
        return self.model(src, src_length)

    def learn(self, act_prob, action, reward, length=None):
        """
        update policy model self.model with policy gradient algorithm
        """
        neg_log_prob = layers.cross_entropy(act_prob, action)
        cost = neg_log_prob * reward
        cost = (layers.reduce_sum(cost) / layers.reduce_sum(length)
                ) if length is not None else layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost


def reward_func(samples, sample_length):
    samples = np.array(samples)
    sample_length = np.array(sample_length)
    reward = (10 - np.abs(sample_length - 10)).astype("float32")
    return discount_reward(reward, sample_length, discount=1.).astype("float32")


def discount_reward(reward, sequence_length, discount=1.):
    return discount_reward_1d(reward, sequence_length, discount)


def discount_reward_1d(reward, sequence_length, discount=1., dtype=None):
    if sequence_length is None:
        raise ValueError('sequence_length must not be `None` for 1D reward.')

    reward = np.array(reward)
    sequence_length = np.array(sequence_length)

    batch_size = reward.shape[0]
    max_seq_length = np.max(sequence_length)
    dtype = dtype or reward.dtype

    if discount == 1.:
        dmat = np.ones([batch_size, max_seq_length], dtype=dtype)
    else:
        steps = np.tile(np.arange(max_seq_length), [batch_size, 1])
        mask = np.asarray(steps < (sequence_length - 1)[:, None], dtype=dtype)
        # Make each row = [discount, ..., discount, 1, ..., 1]
        dmat = mask * discount + (1 - mask)
        dmat = np.cumprod(dmat[:, ::-1], axis=1)[:, ::-1]

    disc_reward = dmat * reward[:, None]
    disc_reward = mask_sequences(disc_reward, sequence_length, dtype=dtype)

    return disc_reward


def mask_sequences(sequence, sequence_length, dtype=None, time_major=False):
    sequence = np.array(sequence)
    sequence_length = np.array(sequence_length)

    rank = sequence.ndim
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")
    batch_size = sequence.shape[0]
    max_time = sequence.shape[1]
    dtype = dtype or sequence.dtype

    if time_major:
        sequence = np.transpose(sequence, axes=[1, 0, 2])

    steps = np.tile(np.arange(max_time), [batch_size, 1])
    mask = np.asarray(steps < sequence_length[:, None], dtype=dtype)
    for _ in range(2, rank):
        mask = np.expand_dims(mask, -1)

    sequence = sequence * mask

    if time_major:
        sequence = np.transpose(sequence, axes=[1, 0, 2])

    return sequence