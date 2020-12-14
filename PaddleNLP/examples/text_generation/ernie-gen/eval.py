import os
import ast
import time
import argparse
import logging

import paddle
import paddle.nn as nn
from tqdm import tqdm
from paddle.io import DataLoader
from paddlenlp.transformers import ErnieForGeneration
from paddlenlp.transformers import ErnieTokenizer
from paddlenlp.datasets import Poetry
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import Rouge1, Rouge2
from paddlenlp.utils.log import logger

from encode import convert_example, after_padding
from decode import beam_search_infilling, post_process, greedy_search_infilling

# yapf: disable
parser = argparse.ArgumentParser('seq2seq model with ERNIE')
parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: "+ ", ".join(list(ErnieTokenizer.pretrained_init_configuration.keys())))
parser.add_argument('--max_encode_len', type=int, default=5)
parser.add_argument('--max_decode_len', type=int, default=5)
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument('--beam_width', type=int, default=5)
parser.add_argument('--noise_prob', type=float, default=0.7, help='probability of token be repalced')
parser.add_argument('--length_penalty', type=float, default=1.0)
parser.add_argument('--init_checkpoint', type=str, default=None, help='checkpoint to warm start from')
parser.add_argument('--use_gpu', action='store_true', help='if set, use gpu to excute')

# yapf: enable

args = parser.parse_args()


def evaluate():
    paddle.set_device("gpu" if args.use_gpu else "cpu")

    model = ErnieForGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)

    dev_dataset = Poetry.get_datasets(['dev'])
    attn_id = tokenizer.vocab[
        '[ATTN]'] if '[ATTN]' in tokenizer.vocab else tokenizer.vocab['[MASK]']
    tgt_type_id = model.sent_emb.weight.shape[0] - 1

    trans_func = convert_example(
        tokenizer=tokenizer,
        attn_id=attn_id,
        tgt_type_id=tgt_type_id,
        max_encode_len=args.max_encode_len,
        max_decode_len=args.max_decode_len)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_pids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_sids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_pids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_sids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # attn_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_labels
    ): after_padding(fn(samples))

    dev_dataset = dev_dataset.apply(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_dataset, batch_size=args.batch_size, shuffle=False)
    data_loader = DataLoader(
        dataset=dev_dataset,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    rouge1 = Rouge1(for_chinese=True)
    rouge2 = Rouge2(for_chinese=True)

    if args.init_checkpoint:
        model_state = paddle.load(args.init_checkpoint)
        model.set_state_dict(model_state)

    model.eval()
    vocab = tokenizer.vocab
    eos_id = vocab[tokenizer.sep_token]
    sos_id = vocab[tokenizer.cls_token]
    pad_id = vocab[tokenizer.pad_token]
    unk_id = vocab[tokenizer.unk_token]
    vocab_size = len(vocab)
    evaluated_sentences = []
    reference_sentences = []
    logger.info("Evaluating...")
    for data in tqdm(data_loader):
        (src_ids, src_sids, src_pids, _, _, _, _, _, _, _, _,
         raw_tgt_labels) = data  # never use target when infer
        # Use greedy_search_infilling or beam_search_infilling to get predictions
        output_ids = beam_search_infilling(
            model,
            src_ids,
            src_sids,
            eos_id=eos_id,
            sos_id=sos_id,
            attn_id=attn_id,
            pad_id=pad_id,
            unk_id=unk_id,
            vocab_size=vocab_size,
            max_decode_len=args.max_decode_len,
            max_encode_len=args.max_encode_len,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            tgt_type_id=tgt_type_id)

        for ids in output_ids.tolist():
            ostr = vocab.to_tokens(ids)
            if '[SEP]' in ostr:
                ostr = ostr[:ostr.index('[SEP]')]
            ostr = ''.join(map(post_process, ostr))
            evaluated_sentences.append(ostr)

        for tgt_label_ids in raw_tgt_labels.numpy().tolist():
            ref = vocab.to_tokens(tgt_label_ids)[1:-1]
            reference_sentences.append(''.join(map(post_process, ref)))

    score1 = rouge1.score(evaluated_sentences, reference_sentences)
    score2 = rouge2.score(evaluated_sentences, reference_sentences)

    print("Rouge-1: %.5f ,Rouge-2: %.5f" % (score1 * 100, score2 * 100))


if __name__ == "__main__":
    evaluate()
