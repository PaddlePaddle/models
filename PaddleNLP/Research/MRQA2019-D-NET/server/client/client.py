#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query the MRQA model server to generate predictions.
"""
import argparse
import json
import requests
import time


if __name__ == '__main__':
    parse = argparse.ArgumentParser("")
    parse.add_argument("dataset")
    parse.add_argument("output_file")
    parse.add_argument("port", type=int)
    args = parse.parse_args()

    all_predictions = {}
    contexts = []
    f = open(args.dataset)
    for example in f:
        context = json.loads(example)
        if 'header' in context:
            continue
        contexts.append(context)
    f.close()

    results = {}
    cnt = 0
    for context in contexts:
        cnt += 1
        start = time.time()
        pred = requests.post('http://127.0.0.1:%d' % args.port, json=context)
        result = pred.json()
        results.update(result)
        end=time.time()
        print('----- request cnt: {}, time elapsed: {:.2f} ms -----'.format(cnt, (end - start)*1000))
        for qid, answer in result.items():
            print('{}: {}'.format(qid, answer.encode('utf-8')))
    with open(args.output_file,'w') as f:
        json.dump(results, f, indent=1)

