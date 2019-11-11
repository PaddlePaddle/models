#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT model service
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
import model_wrapper
import argparse


assert len(sys.argv) == 3 or len(sys.argv) == 4, "Usage: python serve.py <model_dir> <port> [process_mode]"
if len(sys.argv) == 3:
    _, model_dir, port = sys.argv
    mode = 'parallel'
else:
    _, model_dir, port, mode = sys.argv

max_batch_size = 5

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
model = model_wrapper.BertModelWrapper(model_dir=model_dir)
server = mrc_service.MRQAService('MRQA service', app.logger)

@app.route('/', methods=['POST'])
def mrqa_service():
    """Description"""
    return server(model, process_mode=mode, max_batch_size=max_batch_size)


if __name__ == '__main__':
    app.run(port=port, debug=False, threaded=False, processes=1)

