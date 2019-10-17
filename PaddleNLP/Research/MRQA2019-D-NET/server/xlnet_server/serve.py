#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XL-NET model service
"""
import json
import sys
import logging
logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
import requests
from flask import Flask
from flask import Response
from flask import request
import server_utils
import wrapper as bert_wrapper

assert len(sys.argv) == 3 or len(sys.argv) == 4, "Usage: python serve.py <model_dir> <port> [process_mode]"
if len(sys.argv) == 3:
    _, model_dir, port = sys.argv
    mode = 'parallel'
else:
    _, model_dir, port, mode = sys.argv

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
bert_model = bert_wrapper.BertModelWrapper(model_dir=model_dir)
server = server_utils.BasicMRCService('Short answer MRC service', app.logger)

@app.route('/', methods=['POST'])
def mrqa_service():
    """Description"""
    model = bert_model
    return server(model, process_mode=mode, max_batch_size=5)
    # return server(model)


if __name__ == '__main__':
    app.run(port=port, debug=False, threaded=False, processes=1)

