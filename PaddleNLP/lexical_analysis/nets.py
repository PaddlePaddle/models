"""
The function lex_net(args) define the lexical analysis network structure
"""
import sys
import os
import math
import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer


def lex_net(word, args, vocab_size, num_labels, for_infer = True, target=None):
    """
    define the lexical analysis network structure
    word: stores the input of the model
    for_infer: a boolean value, indicating if the model to be created is for training or predicting.

    return:
        for infer: return the prediction
        otherwise: return the prediction
    """
    word_emb_dim = args.word_emb_dim
    grnn_hidden_dim = args.grnn_hidden_dim
    emb_lr = args.emb_learning_rate if 'crf_learning_rate' in dir(args) else 1.0
    crf_lr = args.emb_learning_rate if 'crf_learning_rate' in dir(args) else 1.0
    bigru_num = args.bigru_num
    init_bound = 0.1
    IS_SPARSE = True

    def _bigru_layer(input_feature):
        """
        define the bidirectional gru layer
        """
        pre_gru = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru = fluid.layers.dynamic_gru(
            input=pre_gru,
            size=grnn_hidden_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        pre_gru_r = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru_r = fluid.layers.dynamic_gru(
            input=pre_gru_r,
            size=grnn_hidden_dim,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
        return bi_merge

    def _net_conf(word, target=None):
        """
        Configure the network
        """
        # ipdb.set_trace()
        word_embedding = fluid.layers.embedding(
            input=word,
            size=[vocab_size, word_emb_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound)))

        input_feature = word_embedding
        for i in range(bigru_num):
            bigru_output = _bigru_layer(input_feature)
            input_feature = bigru_output

        emission = fluid.layers.fc(
            size=num_labels,
            input=bigru_output,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        if not for_infer:
            crf_cost = fluid.layers.linear_chain_crf(
                input=emission,
                label=target,
                param_attr=fluid.ParamAttr(
                    name='crfw',
                    learning_rate=crf_lr))
            avg_cost = fluid.layers.mean(x=crf_cost)
            crf_decode = fluid.layers.crf_decoding(
                input=emission, param_attr=fluid.ParamAttr(name='crfw'))
            return avg_cost,crf_decode

        else:
            size = emission.shape[1]
            fluid.layers.create_parameter(shape = [size + 2, size],
                                          dtype=emission.dtype,
                                          name='crfw')
            crf_decode = fluid.layers.crf_decoding(
                input=emission, param_attr=fluid.ParamAttr(name='crfw'))

        return crf_decode

    if for_infer:
        return _net_conf(word)

    else:
        # assert target != None, "target is necessary for training"
        return _net_conf(word, target)


def create_model(args,  vocab_size, num_labels, mode = 'train'):
    """create lac model"""

    # model's input data
    words = fluid.layers.data(name='words', shape=[-1, 1], dtype='int64',lod_level=1)
    targets = fluid.layers.data(name='targets', shape=[-1, 1], dtype='int64', lod_level= 1)

    # for inference process
    if mode=='infer':
        crf_decode = lex_net(words, args, vocab_size, num_labels, for_infer=True, target=None)
        return { "feed_list":[words],"words":words, "crf_decode":crf_decode,}

    # for test or train process
    avg_cost, crf_decode = lex_net(words, args, vocab_size, num_labels, for_infer=False, target=targets)

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
        input=crf_decode,
        label=targets,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((num_labels - 1) / 2.0)))
    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()

    ret = {
        "feed_list":[words, targets],
        "words": words,
        "targets": targets,
        "avg_cost":avg_cost,
        "crf_decode": crf_decode,
        "precision" : precision,
        "recall": recall,
        "f1_score": f1_score,
        "chunk_evaluator": chunk_evaluator,
        "num_infer_chunks": num_infer_chunks,
        "num_label_chunks": num_label_chunks,
        "num_correct_chunks": num_correct_chunks
    }
    return  ret

import paddle
sys.path.append("..")
from models.representation.ernie import ernie_encoder
from preprocess.ernie import task_reader
from reader import Dataset

def create_pyreader(args, file_name, feed_list, place, mode='lac', reader=None, iterable=True, return_reader=False, for_test=False):
    # init reader
    pyreader = fluid.io.PyReader(
        feed_list=feed_list,
        capacity=300,
        use_double_buffer=True,
        iterable=iterable
    )
    if mode == 'lac':
        if reader==None:
            reader = Dataset(args)
        # create lac pyreader
        if for_test:
            pyreader.decorate_sample_list_generator(
                paddle.batch(
                    reader.file_reader(file_name),
                    batch_size=args.batch_size
                ),
                places=place
            )
        else:
            pyreader.decorate_sample_list_generator(
                paddle.batch(
                    paddle.reader.shuffle(
                        reader.file_reader(file_name),
                        buf_size=args.traindata_shuffle_buffer
                    ),
                    batch_size=args.batch_size
                ),
                places=place
            )

    elif mode == 'ernie':
        # create ernie pyreader
        if reader==None:
            reader = task_reader.SequenceLabelReader(
                vocab_path=args.vocab_path,
                label_map_config=args.label_map_config,
                max_seq_len=args.max_seq_len,
                do_lower_case=args.do_lower_case,
                in_tokens=False,
                random_seed=args.random_seed)

        if for_test:
            pyreader.decorate_batch_generator(
                reader.data_generator(
                    file_name, args.batch_size, epoch=1, shuffle=False, phase='test'
                ),
                places=place
            )
        else:
            pyreader.decorate_batch_generator(
                reader.data_generator(
                    file_name, args.batch_size, args.epoch, shuffle=True, phase="train"
                ),
                places=place
            )

    if return_reader:
        return pyreader, reader
    else:
        return pyreader

def create_ernie_model(args,
                 # embeddings,
                 # labels,
                 ernie_config,
                 is_prediction=False):

    """
    Create Model for LAC based on ERNIE encoder
    """
    # ERNIE's input data

    src_ids = fluid.layers.data(name='src_ids', shape=[args.max_seq_len, 1], dtype='int64',lod_level=0)
    sent_ids = fluid.layers.data(name='sent_ids', shape=[args.max_seq_len, 1], dtype='int64',lod_level=0)
    pos_ids = fluid.layers.data(name='pos_ids', shape=[args.max_seq_len, 1], dtype='int64',lod_level=0)
    input_mask = fluid.layers.data(name='input_mask', shape=[args.max_seq_len, 1], dtype='int64',lod_level=0)
    padded_labels =fluid.layers.data(name='padded_labels', shape=[args.max_seq_len, 1], dtype='int64',lod_level=0)
    seq_lens = fluid.layers.data(name='seq_lens', shape=[1], dtype='int64',lod_level=0)

    ernie_inputs = {
        "src_ids": src_ids,
        "sent_ids": sent_ids,
        "pos_ids": pos_ids,
        "input_mask": input_mask,
        "seq_lens": seq_lens
    }
    embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)

    words = fluid.layers.sequence_unpad(src_ids, seq_lens)
    labels = fluid.layers.sequence_unpad(padded_labels, seq_lens)


    # sentence_embeddings = embeddings["sentence_embeddings"]
    token_embeddings = embeddings["token_embeddings"]

    emission = fluid.layers.fc(
        size=args.num_labels,
        input=token_embeddings,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-args.init_bound, high=args.init_bound),
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=1e-4)))


    if is_prediction:
        size = emission.shape[1]
        fluid.layers.create_parameter(shape=[size + 2, size],
                                      dtype=emission.dtype,
                                      name='crfw')
        crf_decode = fluid.layers.crf_decoding(
            input=emission, param_attr=fluid.ParamAttr(name='crfw'))
        ret= {
            "feed_list": [src_ids, sent_ids, pos_ids, input_mask, seq_lens],
            "crf_decode":crf_decode}

    else:
        crf_cost = fluid.layers.linear_chain_crf(
            input=emission,
            label=labels,
            param_attr=fluid.ParamAttr(
                name='crfw',
                learning_rate=args.crf_learning_rate))
        avg_cost = fluid.layers.mean(x=crf_cost)
        crf_decode = fluid.layers.crf_decoding(
            input=emission, param_attr=fluid.ParamAttr(name='crfw'))


        (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
         num_correct_chunks) = fluid.layers.chunk_eval(
             input=crf_decode,
             label=labels,
             chunk_scheme="IOB",
             num_chunk_types=int(math.ceil((args.num_labels - 1) / 2.0)))
        chunk_evaluator = fluid.metrics.ChunkEvaluator()
        chunk_evaluator.reset()

        ret = {
            "feed_list": [src_ids, sent_ids, pos_ids, input_mask, padded_labels, seq_lens],
            "words":words,
            "labels":labels,
            "avg_cost":avg_cost,
            "crf_decode":crf_decode,
            "precision" : precision,
            "recall": recall,
            "f1_score": f1_score,
            "chunk_evaluator":chunk_evaluator,
            "num_infer_chunks":num_infer_chunks,
            "num_label_chunks":num_label_chunks,
            "num_correct_chunks":num_correct_chunks
        }

    return ret
