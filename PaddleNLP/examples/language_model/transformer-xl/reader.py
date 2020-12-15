import os, sys
import glob

from collections import Counter, OrderedDict
import numpy as np

from utils.vocabulary import Vocab

import paddle
from paddle.io import IterableDataset, DataLoader
import paddle.distributed as dist


class LMDataset(IterableDataset):
    def __init__(self, mode, vocab, path, dataset_name, batch_size, bptt,
                 ext_len, nranks, rank):
        assert (mode in ["train", "valid", "test"]
                ), "Parameter mode must be one of [train, valid, test]."

        super(LMDataset, self).__init__()
        self.vocab = vocab
        self.dataset_name = dataset_name

        if self.dataset_name in ["wt2", "wt103"]:
            self.data = self.vocab.encode_file(
                os.path.join(path, mode + ".txt"), ordered=True)
        elif self.dataset_name in ["enwik8", "text8"]:
            self.data = self.vocab.encode_file(
                os.path.join(path, mode + ".txt"), ordered=True, add_eos=False)
        else:
            raise ValueError("Not supported dataset yet. ")
        self.rank = rank
        self.batch_size = batch_size
        batch_size *= nranks

        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.num_step = len(self.data) // batch_size
        data = self.data[:self.num_step * batch_size]
        self.data = data.reshape([batch_size, -1])

        # Number of samples
        self.num_samples = (self.num_step + self.bptt - 1) // self.bptt

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for i in range(0, self.data.shape[1] - 1, self.bptt):
            seq_len = min(self.bptt, self.data.shape[1] - 1 - i)
            end_idx = i + seq_len
            beg_idx = max(0, i - self.ext_len)
            src = self.data[:, beg_idx:end_idx]
            target = self.data[:, i + 1:i + 1 + seq_len]

            # NOTE: `seq_len` will be transfered to numpy immediately
            # after returned by DataLoader. Hence, `seq_len` can be
            # yield as `int`. And the returned tensor `seq_len`'s shape
            # will be empty [].
            # However, if it's necessary to use `seq_len` as input for some
            # PaddlePaddle op, then it must be returned by `[seq_len]` whose
            # shape is [1], cause some op cannot use shape [] as input. 
            yield [
                src[self.rank * self.batch_size:(self.rank + 1) *
                    self.batch_size], target[self.rank * self.batch_size:(
                        self.rank + 1) * self.batch_size], seq_len
            ]


def get_lm_data_loader(args, vocab, mode="train"):
    lm_dataset = LMDataset(
        mode=mode,
        vocab=vocab,
        path=args.data,
        dataset_name=args.dataset,
        batch_size=args.batch_size if mode == "train" else args.eval_batch_size,
        bptt=args.tgt_len,
        ext_len=args.ext_len,
        nranks=dist.get_world_size() if mode == "train" else 1,
        rank=dist.get_rank() if mode == "train" else 0)

    data_loader = DataLoader(
        dataset=lm_dataset, batch_size=None, num_workers=0, return_list=True)

    return data_loader


def get_lm_vocab(args):
    kwargs = {}
    if args.token_delimiter == "None":
        kwargs["delimiter"] = None
    else:
        kwargs["delimiter"] = args.token_delimiter

    if args.dataset in ["wt103", "wt2"]:
        kwargs["special"] = ["<eos>"]
        kwargs["lower_case"] = False
    elif args.dataset == "lm1b":
        kwargs["special"] = []
        kwargs["lower_case"] = False
        kwargs["vocab_file"] = os.path.join(args.data, "1b_word_vocab.txt")

    vocab = Vocab(**kwargs)

    if args.dataset in ["wt2", "enwik8", "text8"]:
        vocab.cnt_file(os.path.join(args.data, "train.txt"))
        vocab.cnt_file(os.path.join(args.data, "valid.txt"))
        vocab.cnt_file(os.path.join(args.data, "test.txt"))
    elif args.dataset == "wt103":
        vocab.cnt_file(os.path.join(args.data, "train.txt"))
    else:
        raise ValueError("Not supported dataset yet. ")

    vocab.build_vocab()
    args.ntokens = len(vocab)

    return vocab
