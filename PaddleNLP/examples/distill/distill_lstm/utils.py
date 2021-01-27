import numpy as np
import jieba
from run_bert_finetune import convert_example


def convert_small_example(example,
                          task_name,
                          vocab,
                          is_tokenized=False,
                          max_seq_length=128,
                          is_test=False):
    input_ids = []
    if task_name == 'senta':
        for i, token in enumerate(jieba.cut(example[0])):
            if i == max_seq_length:
                break
            token_id = vocab[token]
            input_ids.append(token_id)
    else:
        if is_tokenized:
            tokens = example[0][:max_seq_length]
        else:
            tokens = vocab(example[0])[:max_seq_length]
        input_ids = vocab.convert_tokens_to_ids(tokens)

    valid_length = np.array(len(input_ids), dtype='int64')

    if not is_test:
        label = np.array(example[-1], dtype="int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


def convert_pair_example(example,
                         task_name,
                         vocab,
                         is_tokenized=True,
                         max_seq_length=128,
                         is_test=False):
    is_tokenized &= (task_name != 'senta')
    seq1 = convert_small_example([example[0], example[2]], task_name, vocab,
                                 is_tokenized, max_seq_length, is_test)[:2]

    seq2 = convert_small_example([example[1], example[2]], task_name, vocab,
                                 is_tokenized, max_seq_length, is_test)
    pair_features = seq1 + seq2

    return pair_features


def convert_two_example(example,
                        task_name,
                        tokenizer,
                        label_list,
                        max_seq_length,
                        vocab,
                        is_tokenized=True,
                        is_test=False):
    is_tokenized &= (task_name != 'senta')
    bert_features = convert_example(
        example,
        tokenizer=tokenizer,
        label_list=label_list,
        is_tokenized=is_tokenized,
        max_seq_length=max_seq_length,
        is_test=is_test)
    if task_name == 'qqp' or task_name == 'mnli':
        small_features = convert_pair_example(
            example, task_name, vocab, is_tokenized, max_seq_length, is_test)
    else:
        small_features = convert_small_example(
            example, task_name, vocab, is_tokenized, max_seq_length, is_test)

    return bert_features[:2] + small_features
