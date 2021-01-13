import numpy as np
import jieba
from run_bert_finetune import convert_example


def convert_small_example(example,
                          vocab,
                          language='en',
                          max_seq_length=128,
                          is_test=False):
    input_ids = []
    if language == 'cn':
        for i, token in enumerate(jieba.cut(example[0])):
            if i == max_seq_length:
                break
            token_id = vocab[token]
            input_ids.append(token_id)
    else:
        tokens = vocab(example[0])[:max_seq_length]
        input_ids = vocab.convert_tokens_to_ids(tokens)
    valid_length = np.array(len(input_ids), dtype='int64')

    if not is_test:
        label = np.array(example[-1], dtype="int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


def convert_two_example(example,
                        tokenizer,
                        label_list,
                        max_seq_length,
                        vocab,
                        language='en',
                        is_test=False):
    bert_features = convert_example(
        example,
        tokenizer=tokenizer,
        label_list=label_list,
        max_seq_length=max_seq_length,
        is_test=is_test)

    small_features = convert_small_example(example, vocab, language,
                                           max_seq_length, is_test)
    return bert_features[:2] + small_features


def convert_pair_example(example,
                         vocab,
                         language,
                         max_seq_length=128,
                         is_test=False):
    seq1 = convert_small_example([example[0], example[2]], vocab, language,
                                 max_seq_length, is_test)[:2]

    seq2 = convert_small_example([example[1], example[2]], vocab, language,
                                 max_seq_length, is_test)
    pair_features = seq1 + seq2
    return pair_features
