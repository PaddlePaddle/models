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
parser.add_argument('--max_encode_len', type=int, default=24, help="the max encoding sentence length")
parser.add_argument('--max_decode_len', type=int, default=72, help="the max decoding sentence length")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument('--beam_width', type=int, default=1, help="beam search width")
parser.add_argument('--length_penalty', type=float, default=1.0, help="The length penalty during decoding")
parser.add_argument('--init_checkpoint', type=str, default=None, help='checkpoint to warm start from')
parser.add_argument('--use_gpu', action='store_true', help='if set, use gpu to excute')
# yapf: enable

args = parser.parse_args()


def predict():
    paddle.set_device("gpu" if args.use_gpu else "cpu")

    model = ErnieForGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)

    test_dataset = Poetry.get_datasets(['test'])
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

    test_dataset = test_dataset.apply(trans_func, lazy=True)
    test_batch_sampler = paddle.io.BatchSampler(
        test_dataset, batch_size=args.batch_size, shuffle=False)
    data_loader = DataLoader(
        dataset=test_dataset,
        batch_sampler=test_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

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
    evaluated_sentences_ids = []
    logger.info("Predicting...")
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
            if eos_id in ids:
                ids = ids[:ids.index(eos_id)]
            evaluated_sentences_ids.append(ids)

        break
    for ids in evaluated_sentences_ids[:5]:
        evaluated_sentences.append(''.join(
            map(post_process, vocab.to_tokens(ids))))

    for sentence in evaluated_sentences:
        print(sentence)


if __name__ == "__main__":
    predict()
