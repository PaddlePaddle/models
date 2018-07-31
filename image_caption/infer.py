

from __future__ import print_function
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as pd
from paddle.fluid.executor import Executor
import os

dict_size = 4528
source_dict_dim = target_dict_dim = dict_size
hidden_dim = 32
word_dim = 50
batch_size = 2
max_length = 8
topk_size = 50
beam_size = 2

is_sparse = True
decoder_size = hidden_dim
model_save_dir = "machine_translation.inference.model"


def encoder():

    image = pd.data(
        name="image", shape=config.shape, dtype='float32')
    out = encoder.resnet(image)
    return out


def decode(context):
    init_state = context
    array_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = pd.zeros(shape=[1], dtype='int64', force_cpu=True)

    # fill the first element with init_state
    state_array = pd.create_array('float32')
    pd.array_write(init_state, array=state_array, i=counter)

    # ids, scores as memory
    ids_array = pd.create_array('int64')
    scores_array = pd.create_array('float32')

    init_ids = pd.data(name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = pd.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)

    pd.array_write(init_ids, array=ids_array, i=counter)
    pd.array_write(init_scores, array=scores_array, i=counter)

    cond = pd.less_than(x=counter, y=array_len)

    while_op = pd.While(cond=cond)
    with while_op.block():
        pre_ids = pd.array_read(array=ids_array, i=counter)
        pre_state = pd.array_read(array=state_array, i=counter)
        pre_score = pd.array_read(array=scores_array, i=counter)

        # expand the lod of pre_state to be the same with pre_score
        pre_state_expanded = pd.sequence_expand(pre_state, pre_score)

        pre_ids_emb = pd.embedding(
            input=pre_ids,
            size=[dict_size, word_dim],
            dtype='float32',
            is_sparse=is_sparse,
            param_attr=fluid.ParamAttr(name='vemb'))

        # use rnn unit to update rnn
        current_state = pd.fc(
            input=[pre_state_expanded, pre_ids_emb],
            size=decoder_size,
            act='tanh')
        current_state_with_lod = pd.lod_reset(x=current_state, y=pre_score)
        # use score to do beam search
        current_score = pd.fc(
            input=current_state_with_lod, size=target_dict_dim, act='softmax')
        topk_scores, topk_indices = pd.topk(current_score, k=beam_size)
        # calculate accumulated scores after topk to reduce computation cost
        accu_scores = pd.elementwise_add(
            x=pd.log(topk_scores), y=pd.reshape(pre_score, shape=[-1]), axis=0)
        selected_ids, selected_scores = pd.beam_search(
            pre_ids,
            pre_score,
            topk_indices,
            accu_scores,
            beam_size,
            end_id=10,
            level=0)

        with pd.Switch() as switch:
            with switch.case(pd.is_empty(selected_ids)):
                pd.fill_constant(
                    shape=[1], value=0, dtype='bool', force_cpu=True, out=cond)
            with switch.default():
                pd.increment(x=counter, value=1, in_place=True)

                # update the memories
                pd.array_write(current_state, array=state_array, i=counter)
                pd.array_write(selected_ids, array=ids_array, i=counter)
                pd.array_write(selected_scores, array=scores_array, i=counter)

                # update the break condition: up to the max length or all candidates of
                # source sentences have ended.
                length_cond = pd.less_than(x=counter, y=array_len)
                finish_cond = pd.logical_not(pd.is_empty(x=selected_ids))
                pd.logical_and(x=length_cond, y=finish_cond, out=cond)

    translation_ids, translation_scores = pd.beam_search_decode(
        ids=ids_array, scores=scores_array, beam_size=beam_size, end_id=10)

    return translation_ids, translation_scores


def decode_main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    context = encoder()
    translation_ids, translation_scores = decode(context)
    fluid.io.load_persistables(executor=exe, dirname=model_save_dir)

    init_ids_data = np.array([1 for _ in range(batch_size)], dtype='int64')
    init_scores_data = np.array(
        [1. for _ in range(batch_size)], dtype='float32')
    init_ids_data = init_ids_data.reshape((batch_size, 1))
    init_scores_data = init_scores_data.reshape((batch_size, 1))
    init_lod = [1] * batch_size
    init_lod = [init_lod, init_lod]

    init_ids = fluid.create_lod_tensor(init_ids_data, init_lod, place)
    init_scores = fluid.create_lod_tensor(init_scores_data, init_lod, place)

    test_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.test(dict_size), buf_size=1000),
        batch_size=batch_size)

    feed_order = ['src_word_id']
    feed_list = [
        framework.default_main_program().global_block().var(var_name)
        for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)

    for data in test_data():
        feed_data = map(lambda x: [x[0]], data)
        feed_dict = feeder.feed(feed_data)
        feed_dict['init_ids'] = init_ids
        feed_dict['init_scores'] = init_scores

        results = exe.run(
            framework.default_main_program(),
            feed=feed_dict,
            fetch_list=[translation_ids, translation_scores],
            return_numpy=False)

        result_ids = np.array(results[0])
        result_scores = np.array(results[1])

        print("image names:")
        print(" ".join([src_dict[w] for w in feed_data[0][0][1:-1]]))
        print("Translated score and sentence:")
        for i in xrange(beam_size):
            start_pos = result_ids_lod[1][i] + 1
            end_pos = result_ids_lod[1][i + 1]
            print("%d\t%.4f\t%s\n" % (
                i + 1, result_scores[end_pos - 1],
                " ".join([trg_dict[w] for w in result_ids[start_pos:end_pos]])))

        break


def main(use_cuda):
    decode_main(False)  # Beam Search does not support CUDA


if __name__ == '__main__':
    use_cuda = os.getenv('WITH_GPU', '0') != '0'
    main(use_cuda)
