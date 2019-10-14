#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERNIE model service
"""
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
import mrc_service
import model_wrapper as ernie_wrapper

assert len(sys.argv) == 3 or len(sys.argv) == 4, "Usage: python serve.py <model_dir> <port> [process_mode]"
if len(sys.argv) == 3:
    _, model_dir, port = sys.argv
    mode = 'parallel'
else:
    _, model_dir, port, mode = sys.argv

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
ernie_model = ernie_wrapper.ERNIEModelWrapper(model_dir=model_dir)
server = mrc_service.BasicMRCService('Short answer MRC service', app.logger)

@app.route('/', methods=['POST'])
def mrqa_service():
    """Description"""
    model = ernie_model
    return server(model, process_mode=mode, max_batch_size=5)


if __name__ == '__main__':
    app.run(port=port, debug=False, threaded=False, processes=1)

