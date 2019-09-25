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
from multiprocessing.dummy import Pool as ThreadPool

app = Flask(__name__)

logger = logging.getLogger('flask')
url_1 = 'http://127.0.0.1:5118'   # url for model1
url_2 = 'http://127.0.0.1:5120'   # url for model2

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

        pool = ThreadPool(2)
        res1 = pool.apply_async(_call_model, (url_1, input_json))
        res2 = pool.apply_async(_call_model, (url_2, input_json))
        nbest1 = res1.get()
        nbest2 = res2.get()
        # print(res1)
        # print(nbest1)
        pool.close()
        pool.join()

        nbest1 = nbest1.json()['results']
        nbest2 = nbest2.json()['results']
        qids = list(nbest1.keys())
        for qid in qids:
            ensemble_nbest = ensemble_example([nbest1[qid], nbest2[qid]], n_models=2)
            pred[qid] = ensemble_nbest[0]['text']
    except Exception as e:
        pred['error'] = 'empty'
        # logger.error('Error in mrc server - {}'.format(e))
        logger.exception(e)

    # import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return Response(json.dumps(pred), mimetype='application/json')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5121, debug=False, threaded=False, processes=1)

