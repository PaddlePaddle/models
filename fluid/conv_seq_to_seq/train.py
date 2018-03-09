import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.layers as pd
import paddle.fluid.nets as nets
import paddle.fluid.framework as framework
from paddle.fluid.framework import framework_pb2
from utils import *
from model import ConvEncoder, ConvDecoder
import config

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

    encoder = ConvEncoder(
        config.encoder.dict_size,
        config.encoder.word_dim,
        max_positions=config.max_len,
        convolutions=config.encoder.convolutions,
        pad_id=config.encoder.pad_id,
        pos_pad_id=config.max_len,
    )

    encoder_out = encoder.forward(src_tokens, src_positions)

    decoder = ConvDecoder(
        config.decoder.dict_size,
        config.decoder.word_dim,
        config.decoder.word_dim,
        config.max_len,
        convolutions=config.decoder.convolutions,
        pad_id=config.decoder.pad_id,
        pos_pad_id=config.max_len,
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
            paddle.dataset.wmt14.train(config.encoder.dict_size),
            buf_size=1000),
        batch_size=config.batch_size)
    exe = fluid.Executor(place)
    exe.run(fluid.framework.default_startup_program())

    one_batch_data = train_data().next()

    def learn_one_batch(data):
        source = [x[0] for x in data]
        source_pos = [[i for i in range(len(x))] for x in source]
        # TODO need to process the pre
        target_pre = [x[1] for x in data]
        #print 'target_pre', target_pre
        target_pre_pos = [[i for i in range(len(x))] for x in target_pre]
        target = [x[1] for x in data]

        src_word = prepare_data(source, config.encoder.start_id,
                                config.encoder.end_id, config.encoder.pad_id,
                                config.max_len)
        src_posi = prepare_data(source_pos, config.max_len, config.max_len,
                                config.max_len, config.max_len)
        target_pre = prepare_data(
            target_pre,
            config.decoder.start_id,
            config.decoder.end_id,
            config.decoder.pad_id,
            config.max_len,
            offset=-1)
        target_pre_pos = prepare_data(target_pre_pos, config.max_len,
                                      config.max_len, config.max_len,
                                      config.max_len)
        target = prepare_data(target, config.decoder.start_id,
                              config.decoder.end_id, config.decoder.pad_id,
                              config.max_len)

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
        return outs

    for pass_id in xrange(config.num_pass):
        if config.mode == 'debug':
            data = one_batch_data
            outs = learn_one_batch(data)
            print 'avg_cost', outs[0]
        elif config.mode == 'train':
            for data in train_data():
                outs = learn_one_batch(data)
                print 'avg_cost', outs[0]


if __name__ == '__main__':
    train_main(config.debug_train_config)
