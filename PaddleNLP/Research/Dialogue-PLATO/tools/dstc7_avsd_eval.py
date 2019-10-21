import sys
import math
import json

import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def_scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(),"METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr")
]

best_scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(),"METEOR"),
    (Rouge(), "ROUGE_L")
]

def score_fn(ref, sample, scorers=def_scorers):
    # ref and sample are both dict
    
    final_scores = {}
    for scorer, method in scorers:
        # print('computing %s score with COCO-EVAL...'%(scorer.method()))
        score, scores = scorer.compute_score(ref, sample)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

from collections import defaultdict
chosen_by_scores = defaultdict(int)
chosen_by_best = defaultdict(int)

acc = 0

with open(sys.argv[1]) as file:
    datas = json.load(file)

cnt = 0
all_refs = dict()
all_cands = dict()

for data in datas:
    ref = list(map(lambda x : x.strip(), data['tgt'].split('|')))

    # if False:
    best_pred = ''
    best_score = -1e9
    best_idx = -1
    for i, pred in enumerate(data['preds']):
        refs = dict()
        cands = dict()
        refs[0] = ref
        cands[0] = [pred]
        ret = score_fn(refs, cands, best_scorers)
        score = sum(map(lambda x : ret[x], ret))
        if score > best_score:
            best_idx = i
            best_score = score
            best_pred = pred
    chosen_by_best[best_idx] += 1

    idx = np.argmax(data['scores'])
    chosen_by_scores[idx] += 1
    chosen_pred = data['preds'][idx]

    if idx == best_idx:
        acc += 1

    all_refs[cnt] = ref
    all_cands[cnt] = [chosen_pred]
    cnt += 1

print(f"Acc: {acc / len(datas)}")
for i in range(20):
    print(f"{i} {chosen_by_scores[i]} {chosen_by_best[i]}"
          f" {chosen_by_scores[i] / len(datas):.4f}"
          f" {chosen_by_scores[i] / chosen_by_best[i]:.4f}")
res = score_fn(all_refs, all_cands)
for name in res:
    print(f"{name}: {res[name]:.4f}")
