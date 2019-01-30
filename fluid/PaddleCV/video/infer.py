#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import time
import logging
import argparse
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import paddle.fluid as fluid

import models

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name',
        type=str,
        default='AttentionCluster',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/attention_cluster.txt',
        help='path to config file of model')
    parser.add_argument(
        '--use-gpu', type=bool, default=True, help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to use weights from Paddle.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--filelist',
        type=str,
        default=None,
        help='path to inferenece data file lists file.')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--infer-topk',
        type=int,
        default=20,
        help='topk predictions to restore.')
    parser.add_argument(
        '--save-dir', type=str, default='./', help='directory to store results')
    args = parser.parse_args()
    return args


def infer(infer_model, args):
    infer_model.build_input(use_pyreader=False)
    infer_model.build_model()
    infer_feeds = infer_model.feeds()
    infer_outputs = infer_model.outputs()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # get infer reader
    if not args.filelist:
        logger.error("[INFER] --filelist unset.")
        return
    assert os.path.exists(args.filelist), "{} not exist.".format(args.filelist)
    infer_reader = infer_model.reader()

    if args.weights:
        assert os.path.exists(
            args.weights), "Given weight dir {} not exist.".format(args.weights)
    # if no weight files specified, download weights from paddle
    weights = args.weights or infer_model.get_weights()

    def if_exist(var):
        return os.path.exists(os.path.join(weights, var.name))

    fluid.io.load_vars(exe, weights, predicate=if_exist)

    infer_feeder = fluid.DataFeeder(place=place, feed_list=infer_feeds)
    fetch_list = [x.name for x in infer_outputs]

    def _infer_loop():
        periods = []
        results = []
        cur_time = time.time()
        for infer_iter, data in enumerate(infer_reader()):
            data_feed_in = [items[:-1] for items in data]
            video_id = [items[-1] for items in data]
            infer_outs = exe.run(fetch_list=fetch_list,
                                 feed=infer_feeder.feed(data_feed_in))
            predictions = np.array(infer_outs[0])
            for i in range(len(predictions)):
                topk_inds = predictions[i].argsort()[0 - args.infer_topk:]
                topk_inds = topk_inds[::-1]
                preds = predictions[i][topk_inds]
                results.append(
                    (video_id[i], preds.tolist(), topk_inds.tolist()))
            prev_time = cur_time
            cur_time = time.time()
            period = cur_time - prev_time
            periods.append(period)
            logger.info('Processed {} samples'.format((infer_iter) * len(
                predictions)))

        logger.info('[INFER] infer finished. average time: {}'.format(
            np.mean(periods)))
        return results

# start infer loop

    infer_results = _infer_loop()
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    result_file_name = os.path.join(args.save_dir,
                                    "{}_infer_result".format(args.model_name))
    pickle.dump(infer_results, open(result_file_name, 'wb'))

if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    infer_model = models.get_model(
        args.model_name, args.config, mode='infer', args=vars(args))
    infer(infer_model, args)
