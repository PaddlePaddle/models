import os
import sys
import re
from collections import defaultdict


def load_dict(word_dict_file):
    word_dict = {}
    with open(word_dict_file, "r") as fin:
        for i, line in enumerate(fin):
            key = line.strip().decode("utf8", errors="ignore").split("\t")[0]
            word_dict[key] = i
    return word_dict


def find_optiaml_pass(log_file):
    cost_info = defaultdict(list)
    cost_pat = re.compile(r'Cost\s[\d]+.[\d]+')
    pass_pat = re.compile(r'Pass\s[\d]+')
    with open(log_file, 'r') as flog:
        for line in flog:
            if not 'Cost' in line: continue
            pass_id = pass_pat.findall(line.strip())[0]
            cost = float(cost_pat.findall(line.strip())[0].replace('Cost ', ''))
            cost_info[pass_id].append(cost)
    print("optimal pass : %s" % sorted(
        cost_info.iteritems(),
        key=lambda x: sum(x[1]) / (len(x[1])),
        reverse=False)[0][0])
