import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.layers as pd
import paddle.fluid.nets as nets
import paddle.fluid.framework as framework
from paddle.fluid.framework import framework_pb2
from utils import *
from model import ConvEncoder, ConvDecoder
import config

MAX_LEN = 40
MAX_LEN_WITH_PAD = MAX_LEN + 1
batch_size = 10
dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
hidden_dim = 32
word_dim = 128
batch_size = 10
max_length = 8
topk_size = 50
trg_dic_size = 10000
beam_size = 2

# special tokens
pad_id = 0  # need to make sure the pad embedding is zero
start_id = 1
end_id = 2
is_test = False


def build_trainer(config):
    src_tokens = pd.data(
        'src_word_id',
        shape=[config.batch_size, config.max_len, 1],
        dtype='int64',
        append_batch_size=False)
    src_positions = pd.data(
        'src_posi_id',
        shape=[config.batch_size, config.max_len, 1],
        dtype='int64',
        append_batch_size=False)
    trg_pre_tokens = pd.data(
        'trg_pre_word_id',
        shape=[config.batch_size, config.max_len, 1],
        dtype='int64',
        append_batch_size=False)
    trg_pre_positions = pd.data(
        'trg_pre_posi_id',
        shape=[config.batch_size, config.max_len, 1],
        dtype='int64',
        append_batch_size=False)
    trg_tokens = pd.data(
        'trg_word_id',
        shape=[config.batch_size, config.max_len, 1],
        dtype='int64',
        append_batch_size=False)

    embed_dim = word_dim
    max_positions = MAX_LEN
    encoder = ConvEncoder(
        dict_size,
        embed_dim,
        max_positions=max_positions,
        convolutions=config.encoder.convolutions,
        pad_id=pad_id,
    )

    encoder_out = encoder.forward(src_tokens, src_positions)

    out_embed_dim = embed_dim
    decoder = ConvDecoder(
        dict_size,
        embed_dim,
        out_embed_dim,
        max_positions,
        convolutions=config.decoder.convolutions,
        pad_id=pad_id,
        attention=True,
    )

    predictions, _ = decoder.forward(trg_pre_tokens, trg_pre_positions,
                                     encoder_out)
    predictions = Op.reshape(predictions,
                             [-1] + [list(get_dims(predictions))[-1]])
    trg_tokens = Op.reshape(trg_tokens, [-1, 1])
    cost = pd.cross_entropy(input=predictions, label=trg_tokens)
    avg_cost = pd.mean(cost)

    optimizer = fluid.optimizer.Adagrad(learning_rate=1e-3)
    optimize_ops, params_grads = optimizer.minimize(avg_cost)
    return avg_cost, predictions


def train_main(config):
    place = fluid.CPUPlace()

    touts = build_trainer(config)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)
    exe = fluid.Executor(place)
    exe.run(fluid.framework.default_startup_program())

    one_batch_data = train_data().next()

    for pass_id in xrange(1000):
        # for data in train_data():
        data = one_batch_data
        source = map(lambda x: x[0], data)
        source_pos = [[i for i in range(len(x))] for x in source]
        # TODO need to process the pre
        target_pre = map(lambda x: x[1], data)
        target_pre_pos = [[i for i in range(len(x))] for x in target_pre]
        target = map(lambda x: x[1], data)

        src_word = to_tensor(source, MAX_LEN)
        src_posi = to_tensor(source_pos, MAX_LEN)
        target_pre = to_tensor(target_pre, MAX_LEN)
        target_pre_pos = to_tensor(target_pre_pos, MAX_LEN)
        target = to_tensor(target, MAX_LEN)

        outs = exe.run(
            framework.default_main_program(),
            feed={
                'src_word_id': src_word,
                'src_posi_id': src_posi,
                'trg_pre_word_id': target_pre,
                'trg_pre_posi_id': target_pre_pos,
                'trg_word_id': target,
            },
            fetch_list=[touts[0], touts[1]])
        print 'avg_cost', outs[0]


if __name__ == '__main__':
    train_main(config.debug_train_config)
