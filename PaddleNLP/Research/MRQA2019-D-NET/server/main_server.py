#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
import requests
from flask import Flask
from flask import Response
from flask import request
import numpy as np
import argparse
from multiprocessing.dummy import Pool as ThreadPool

app = Flask(__name__)

logger = logging.getLogger('flask')


def ensemble_example(answers, n_models=None):
    if n_models is None:
        n_models = len(answers)
    answer_dict = dict()
    for nbest_predictions in answers:
        for prediction in nbest_predictions:
            score_list = answer_dict.setdefault(prediction['text'], [])
            score_list.append(prediction['probability'])

    ensemble_nbest_predictions = []
    for answer, scores in answer_dict.items():
        prediction = dict()
        prediction['text'] = answer
        prediction['probability'] = np.sum(scores) / n_models
        ensemble_nbest_predictions.append(prediction)

    ensemble_nbest_predictions = \
        sorted(ensemble_nbest_predictions, key=lambda item: item['probability'], reverse=True)
    return ensemble_nbest_predictions


@app.route('/', methods=['POST'])
def mrqa_main():
    """Description"""
    # parse input data
    pred = {}
    def _call_model(url, input_json):
        nbest = requests.post(url, json=input_json)
        return nbest
    try:
        input_json = request.get_json(silent=True)
        n_models = len(urls)
        pool = ThreadPool(n_models)
        results = []
        for url in urls:
            result = pool.apply_async(_call_model, (url, input_json))
            results.append(result.get())
        pool.close()
        pool.join()
        nbests = [nbest.json()['results'] for nbest in results]
        qids = list(nbests[0].keys())
        for qid in qids:
            ensemble_nbest = ensemble_example([nbest[qid] for nbest in nbests], n_models=n_models)
            pred[qid] = ensemble_nbest[0]['text']
    except Exception as e:
        pred['error'] = 'empty'
        logger.exception(e)

    return Response(json.dumps(pred), mimetype='application/json')


if __name__ == '__main__':
    url_1 = 'http://127.0.0.1:5118'   # url for ernie
    url_2 = 'http://127.0.0.1:5119'   # url for xl-net
    url_3 = 'http://127.0.0.1:5120'   # url for bert
    parser = argparse.ArgumentParser('main server')
    parser.add_argument('--ernie', action='store_true', default=False, help="Include ERNIE")
    parser.add_argument('--xlnet', action='store_true', default=False, help="Include XL-NET")
    parser.add_argument('--bert', action='store_true', default=False, help="Include BERT")
    args = parser.parse_args()
    urls = []
    if args.ernie:
        print('Include ERNIE model')
        urls.append(url_1)
    if args.xlnet:
        print('Include XL-NET model')
        urls.append(url_2)
    if args.bert:
        print('Include BERT model')
        urls.append(url_3)
    assert len(urls) > 0, "At lease one model is required"
    app.run(host='127.0.0.1', port=5121, debug=False, threaded=False, processes=1)

