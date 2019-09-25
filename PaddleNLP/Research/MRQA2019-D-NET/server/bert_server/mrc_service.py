#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Some utilities for MRC online service"""
import json
import sys
import logging
import time
import numpy as np
from flask import Response
from flask import request
from copy import deepcopy

verbose = False

def _request_check(input_json):
    """Check if the request json is valid"""
    if input_json is None or not isinstance(input_json, dict):
        return 'Can not parse the input json data - {}'.format(input_json)
    try:
        c = input_json['context']
        qa = input_json['qas'][0]
        qid = qa['qid']
        q = qa['question']
    except KeyError as e:
        return 'Invalid request, key "{}" not found'.format(e)
    return 'OK'

def _abort(status_code, message):
    """Create custom error message and status code"""
    return Response(json.dumps(message), status=status_code, mimetype='application/json')

def _timmer(init_start, start, current, process_name):
    cumulated_elapsed_time = (current - init_start) * 1000
    current_elapsed_time = (current - start) * 1000
    print('{}\t-\t{:.2f}\t{:.2f}'.format(process_name, cumulated_elapsed_time,
                                         current_elapsed_time))

def _split_input_json(input_json):
    if len(input_json['context_tokens']) > 810:
        input_json['context'] = input_json['context'][:5000]
    if len(input_json['qas']) == 1:
        return [input_json]
    else:
        rets = []
        for i in range(len(input_json['qas'])):
            temp = deepcopy(input_json)
            temp['qas'] = [input_json['qas'][i]]
            rets.append(temp)
        return rets
            
class MRQAService(object):
    """Provide basic MRC service for flask"""
    def __init__(self, name, logger=None, log_data=False):
        """ """
        self.name = name
        if logger is None:
            self.logger = logging.getLogger('flask')
        else:
            self.logger = logger
        self.log_data = log_data

    def __call__(self, model, process_mode='serial', max_batch_size=5, timmer=False):
        """
        Args:
            mode: serial, parallel
        """
        if timmer:
            start = time.time()
        """Call mrc model wrapper and handle expectations"""
        self.input_json = request.get_json(silent=True)
        try:
            if timmer:
                start_request_check = time.time()
            request_status = _request_check(self.input_json)
            if timmer:
                current_time = time.time()
                _timmer(start, start_request_check, current_time, 'request check')
            if self.log_data:
                if self.logger is None:
                    logging.info(
                        'Client input - {}'.format(json.dumps(self.input_json, ensure_ascii=False))
                    )
                else:
                    self.logger.info(
                        'Client input - {}'.format(json.dumps(self.input_json, ensure_ascii=False))
                    )
        except Exception as e:
            self.logger.error('server request checker error')
            self.logger.exception(e)
            return _abort(500, 'server request checker error - {}'.format(e))
        if request_status != 'OK':
            return _abort(400, request_status)

        # call preprocessor
        try:
            if timmer:
                start_preprocess = time.time()

            jsons = _split_input_json(self.input_json)
            processed = []
            ex_start_idx = 0
            feat_start_idx = 1000000000
            for i in jsons:
                e,f,b = model.preprocessor(i, batch_size=max_batch_size if process_mode == 'parallel' else 1, examples_start_id=ex_start_idx, features_start_id=feat_start_idx)
                ex_start_idx += len(e)
                feat_start_idx += len(f)
                processed.append([e,f,b])

            if timmer:
                current_time = time.time()
                _timmer(start, start_preprocess, current_time, 'preprocess')
        except Exception as e:
            self.logger.error('preprocessor error')
            self.logger.exception(e)
            return _abort(500, 'preprocessor error - {}'.format(e))

        def transpose(mat):
            return zip(*mat)
            
        # call mrc
        try:
            if timmer:
                start_call_mrc = time.time()

            self.mrc_results = []
            self.examples = []
            self.features = []
            for e, f, batches in processed:
                if verbose:
                    if len(f) > max_batch_size:
                        print("get a too long example....")
                if process_mode == 'serial':
                    self.mrc_results.extend([model.call_mrc(b, squeeze_dim0=True) for b in batches[:max_batch_size]])
                elif process_mode == 'parallel':
                    # only keep first max_batch_size features
                    # batches = batches[0]

                    for b in batches:
                        self.mrc_results.extend(model.call_mrc(b, return_list=True))
                else:
                    raise NotImplementedError()
                self.examples.extend(e)
                # self.features.extend(f[:max_batch_size])
                self.features.extend(f)

            if timmer:
                current_time = time.time()
                _timmer(start, start_call_mrc, current_time, 'call mrc')
        except Exception as e:
            self.logger.error('call_mrc error')
            self.logger.exception(e)
            return _abort(500, 'call_mrc error - {}'.format(e))

        # call post processor
        try:
            if timmer:
                start_post_precess = time.time()
            self.results = model.postprocessor(self.examples, self.features, self.mrc_results)

            # only nbest results is POSTed back
            self.results = self.results[1]
            # self.results = self.results[0]

            if timmer:
                current_time = time.time()
                _timmer(start, start_post_precess, current_time, 'post process')
        except Exception as e:
            self.logger.error('postprocessor error')
            self.logger.exception(e)
            return _abort(500, 'postprocessor error - {}'.format(e))

        return self._response_constructor()

    def _response_constructor(self):
        """construct http response object"""
        try:
            response = {
                # 'requestID': self.input_json['requestID'],
                'results': self.results
            }
            if self.log_data:
                self.logger.info(
                    'Response - {}'.format(json.dumps(response, ensure_ascii=False))
                )
            return Response(json.dumps(response), mimetype='application/json')
        except Exception as e:
            self.logger.error('response constructor error')
            self.logger.exception(e)
            return _abort(500, 'response constructor error - {}'.format(e))
