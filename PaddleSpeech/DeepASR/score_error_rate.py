from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from tools.error_rate import char_errors, word_errors


def parse_args():
    parser = argparse.ArgumentParser(
        "Score word/character error rate (WER/CER) "
        "for decoding result.")
    parser.add_argument(
        '--error_rate_type',
        type=str,
        default='cer',
        choices=['cer', 'wer'],
        help="Error rate type. (default: %(default)s)")
    parser.add_argument(
        '--special_tokens',
        type=str,
        default='<SPOKEN_NOISE>',
        help="Special tokens in scoring CER, seperated by space. "
        "They shouldn't be splitted and should be treated as one special "
        "character. Example: '<SPOKEN_NOISE> <bos> <eos>' "
        "(default: %(default)s)")
    parser.add_argument(
        '--ref', type=str, required=True, help="The ground truth text.")
    parser.add_argument(
        '--hyp', type=str, required=True, help="The decoding result text.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    ref_dict = {}
    sum_errors, sum_ref_len = 0.0, 0
    sent_cnt, not_in_ref_cnt = 0, 0

    special_tokens = args.special_tokens.split(" ")

    with open(args.ref, "r") as ref_txt:
        line = ref_txt.readline()
        while line:
            del_pos = line.find(" ")
            key, sent = line[0:del_pos], line[del_pos + 1:-1].strip()
            ref_dict[key] = sent
            line = ref_txt.readline()

    with open(args.hyp, "r") as hyp_txt:
        line = hyp_txt.readline()
        while line:
            del_pos = line.find(" ")
            key, sent = line[0:del_pos], line[del_pos + 1:-1].strip()
            sent_cnt += 1
            line = hyp_txt.readline()
            if key not in ref_dict:
                not_in_ref_cnt += 1
                continue

            if args.error_rate_type == 'cer':
                for sp_tok in special_tokens:
                    sent = sent.replace(sp_tok, '\0')
                errors, ref_len = char_errors(
                    ref_dict[key].decode("utf8"),
                    sent.decode("utf8"),
                    remove_space=True)
            else:
                errors, ref_len = word_errors(ref_dict[key].decode("utf8"),
                                              sent.decode("utf8"))
            sum_errors += errors
            sum_ref_len += ref_len

    print("Error rate[%s] = %f (%d/%d)," %
          (args.error_rate_type, sum_errors / sum_ref_len, int(sum_errors),
           sum_ref_len))
    print("total %d sentences in hyp, %d not presented in ref." %
          (sent_cnt, not_in_ref_cnt))
