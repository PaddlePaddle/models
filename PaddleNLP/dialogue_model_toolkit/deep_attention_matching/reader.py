"""
Reader for deep attention matching network
"""

import six
import numpy as np

try:
    import cPickle as pickle  #python 2
except ImportError as e:
    import pickle  #python 3


def unison_shuffle(data, seed=None):
    """
    Shuffle data
    """
    if seed is not None:
        np.random.seed(seed)

    y = np.array(data[six.b('y')])
    c = np.array(data[six.b('c')])
    r = np.array(data[six.b('r')])

    assert len(y) == len(c) == len(r)
    p = np.random.permutation(len(y))
    print(p)
    shuffle_data = {six.b('y'): y[p], six.b('c'): c[p], six.b('r'): r[p]}
    return shuffle_data


def split_c(c, split_id):
    """
    Split
    c is a list, example context
    split_id is a integer, conf[_EOS_]
    return nested list
    """
    turns = [[]]
    for _id in c:
        if _id != split_id:
            turns[-1].append(_id)
        else:
            turns.append([])
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns


def normalize_length(_list, length, cut_type='tail'):
    """_list is a list or nested list, example turns/r/single turn c
       cut_type is head or tail, if _list len > length is used
       return a list len=length and min(read_length, length)
    """
    real_length = len(_list)
    if real_length == 0:
        return [0] * length, 0

    if real_length <= length:
        if not isinstance(_list[0], list):
            _list.extend([0] * (length - real_length))
        else:
            _list.extend([[]] * (length - real_length))
        return _list, real_length

    if cut_type == 'head':
        return _list[:length], length
    if cut_type == 'tail':
        return _list[-length:], length


def produce_one_sample(data,
                       index,
                       split_id,
                       max_turn_num,
                       max_turn_len,
                       turn_cut_type='tail',
                       term_cut_type='tail'):
    """max_turn_num=10
       max_turn_len=50
       return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len
    """
    c = data[six.b('c')][index]
    r = data[six.b('r')][index][:]
    y = data[six.b('y')][index]

    turns = split_c(c, split_id)
    #normalize turns_c length, nor_turns length is max_turn_num
    nor_turns, turn_len = normalize_length(turns, max_turn_num, turn_cut_type)

    nor_turns_nor_c = []
    term_len = []
    #nor_turn_nor_c length is max_turn_num, element is a list length is max_turn_len
    for c in nor_turns:
        #nor_c length is max_turn_len
        nor_c, nor_c_len = normalize_length(c, max_turn_len, term_cut_type)
        nor_turns_nor_c.append(nor_c)
        term_len.append(nor_c_len)

    nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)

    return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len


def build_one_batch(data,
                    batch_index,
                    conf,
                    turn_cut_type='tail',
                    term_cut_type='tail'):
    """
    Build one batch
    """
    _turns = []
    _tt_turns_len = []
    _every_turn_len = []

    _response = []
    _response_len = []

    _label = []

    for i in six.moves.xrange(conf['batch_size']):
        index = batch_index * conf['batch_size'] + i
        y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len = produce_one_sample(
            data, index, conf['_EOS_'], conf['max_turn_num'],
            conf['max_turn_len'], turn_cut_type, term_cut_type)

        _label.append(y)
        _turns.append(nor_turns_nor_c)
        _response.append(nor_r)
        _every_turn_len.append(term_len)
        _tt_turns_len.append(turn_len)
        _response_len.append(r_len)

    return _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label


def build_one_batch_dict(data,
                         batch_index,
                         conf,
                         turn_cut_type='tail',
                         term_cut_type='tail'):
    """
    Build one batch dict
    """
    _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label = build_one_batch(
        data, batch_index, conf, turn_cut_type, term_cut_type)
    ans = {
        'turns': _turns,
        'tt_turns_len': _tt_turns_len,
        'every_turn_len': _every_turn_len,
        'response': _response,
        'response_len': _response_len,
        'label': _label
    }
    return ans


def build_batches(data, conf, turn_cut_type='tail', term_cut_type='tail'):
    """
    Build batches
    """
    _turns_batches = []
    _tt_turns_len_batches = []
    _every_turn_len_batches = []

    _response_batches = []
    _response_len_batches = []

    _label_batches = []

    batch_len = len(data[six.b('y')]) // conf['batch_size']
    for batch_index in six.moves.range(batch_len):
        _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label = build_one_batch(
            data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail')

        _turns_batches.append(_turns)
        _tt_turns_len_batches.append(_tt_turns_len)
        _every_turn_len_batches.append(_every_turn_len)

        _response_batches.append(_response)
        _response_len_batches.append(_response_len)

        _label_batches.append(_label)

    ans = {
        "turns": _turns_batches,
        "tt_turns_len": _tt_turns_len_batches,
        "every_turn_len": _every_turn_len_batches,
        "response": _response_batches,
        "response_len": _response_len_batches,
        "label": _label_batches
    }

    return ans


def make_one_batch_input(data_batches, index):
    """Split turns and return feeding data.

    Args:
        data_batches: All data batches
        index: The index for current batch

    Return:
        feeding dictionary
    """

    turns = np.array(data_batches["turns"][index])
    tt_turns_len = np.array(data_batches["tt_turns_len"][index])
    every_turn_len = np.array(data_batches["every_turn_len"][index])
    response = np.array(data_batches["response"][index])
    response_len = np.array(data_batches["response_len"][index])

    batch_size = turns.shape[0]
    max_turn_num = turns.shape[1]
    max_turn_len = turns.shape[2]

    turns_list = [turns[:, i, :] for i in six.moves.xrange(max_turn_num)]
    every_turn_len_list = [
        every_turn_len[:, i] for i in six.moves.xrange(max_turn_num)
    ]

    feed_list = []
    for i, turn in enumerate(turns_list):
        turn = np.expand_dims(turn, axis=-1)
        feed_list.append(turn)

    for i, turn_len in enumerate(every_turn_len_list):
        turn_mask = np.ones((batch_size, max_turn_len, 1)).astype("float32")
        for row in six.moves.xrange(batch_size):
            turn_mask[row, turn_len[row]:, 0] = 0
        feed_list.append(turn_mask)

    response = np.expand_dims(response, axis=-1)
    feed_list.append(response)

    response_mask = np.ones((batch_size, max_turn_len, 1)).astype("float32")
    for row in six.moves.xrange(batch_size):
        response_mask[row, response_len[row]:, 0] = 0
    feed_list.append(response_mask)

    label = np.array([data_batches["label"][index]]).reshape(
        [-1, 1]).astype("float32")
    feed_list.append(label)

    return feed_list


if __name__ == '__main__':
    conf = {
        "batch_size": 256,
        "max_turn_num": 10,
        "max_turn_len": 50,
        "_EOS_": 28270,
    }
    with open('../ubuntu/data/data_small.pkl', 'rb') as f:
        if six.PY2:
            train, val, test = pickle.load(f)
        else:
            train, val, test = pickle.load(f, encoding="bytes")
    print('load data success')

    train_batches = build_batches(train, conf)
    val_batches = build_batches(val, conf)
    test_batches = build_batches(test, conf)
    print('build batches success')
