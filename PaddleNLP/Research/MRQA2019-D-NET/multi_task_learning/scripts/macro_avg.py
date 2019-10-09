import numpy as np
import argparse
import re

def extract_score(line):
    prog = re.compile(r'{"f1": (-?\d+\.?\d*e?-?\d*?), "exact_match": (-?\d+\.?\d*e?-?\d*?)}')
    result = prog.match(line)
    f1 = float(result.group(1))
    em = float(result.group(2))
    return f1, em


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Calculate macro average for MRQA')
    parser.add_argument('input_file', help='Score file')
    args = parser.parse_args()
    with open(args.input_file) as fin:
        lines = map(str.strip, fin.readlines())
    in_domain_scores = {}
    for dataset_id in range(0, 12, 2):
        f1, em = extract_score(lines[dataset_id+1])
        in_domain_scores[lines[dataset_id]] = f1
    out_of_domain_scores = {}
    for dataset_id in range(12, 24, 2):
        f1, em = extract_score(lines[dataset_id+1])
        out_of_domain_scores[lines[dataset_id]] = f1
    print('In domain avg: {}'.format(np.mean(in_domain_scores.values())))
    print('Out of domain avg: {}'.format(np.mean(out_of_domain_scores.values())))
