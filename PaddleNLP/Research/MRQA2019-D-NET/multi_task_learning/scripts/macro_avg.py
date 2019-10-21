import numpy as np
import argparse
import json
import re

def extract_score(line):
    score_json = json.loads(line)
    f1 = score_json['f1']
    em = score_json['exact_match']
    return float(f1), float(em)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Calculate macro average for MRQA')
    parser.add_argument('input_file', help='Score file')
    args = parser.parse_args()
    with open(args.input_file) as fin:
        lines = list(map(str.strip, fin.readlines()))
    in_domain_scores = {}
    for dataset_id in range(0, 12, 2):
        f1, em = extract_score(lines[dataset_id+1])
        in_domain_scores[lines[dataset_id]] = f1
    out_of_domain_scores = {}
    for dataset_id in range(12, 24, 2):
        f1, em = extract_score(lines[dataset_id+1])
        out_of_domain_scores[lines[dataset_id]] = f1
    print('In domain avg: {}'.format(sum(in_domain_scores.values()) / len(in_domain_scores.values())))
    print('Out of domain avg: {}'.format(sum(out_of_domain_scores.values()) / len(in_domain_scores.values())))
