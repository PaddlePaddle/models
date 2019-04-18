#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Load checkpoint of running classifier to do prediction and save inference model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid

from finetune_args import parser
from utils.args import print_arguments
from utils.init import init_pretraining_params, init_checkpoint

import define_predict_pack
import reader.data_reader as reader

_WORK_DIR = os.path.split(os.path.realpath(__file__))[0]
sys.path.append('../../models/dialogue_model_toolkit/dialogue_general_understanding')

from bert import BertConfig, BertModel 
from create_model import create_model
import define_paradigm 


def main(args):
    """main function"""
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    task_name = args.task_name.lower()
    paradigm_inst = define_paradigm.Paradigm(task_name)
    pred_inst = define_predict_pack.DefinePredict()
    pred_func = getattr(pred_inst, pred_inst.task_map[task_name])

    processors = {
        'udc': reader.UDCProcessor,
        'swda': reader.SWDAProcessor,
        'mrda': reader.MRDAProcessor,
        'atis_slot': reader.ATISSlotProcessor, 
        'atis_intent': reader.ATISIntentProcessor,
        'dstc2': reader.DSTC2Processor, 
        'dstc2_asr': reader.DSTC2Processor,  
    }

    in_tokens = {
        'udc': True,
        'swda': True,
        'mrda': True,
        'atis_slot': False,
        'atis_intent': True,
        'dstc2': True,   
        'dstc2_asr': True  
    }

    processor = processors[task_name](data_dir=args.data_dir,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case, 
                                      in_tokens=in_tokens[task_name],
                                      task_name=task_name, 
                                      random_seed=args.random_seed)
    num_labels = len(processor.get_labels())

    predict_prog = fluid.Program()
    predict_startup = fluid.Program()
    with fluid.program_guard(predict_prog, predict_startup):
        with fluid.unique_name.guard():
            pred_results = create_model(
                args,
                pyreader_name='predict_reader',
                bert_config=bert_config,
                num_labels=num_labels,
                paradigm_inst=paradigm_inst,
                is_prediction=True)
            predict_pyreader = pred_results.get('pyreader', None)
            probs = pred_results.get('probs', None)
            feed_target_names = pred_results.get('feed_target_names', None)

    predict_prog = predict_prog.clone(for_test=True)

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(predict_startup)

    if args.init_checkpoint:
        init_pretraining_params(exe, args.init_checkpoint, predict_prog)
    else:
        raise ValueError("args 'init_checkpoint' should be set for prediction!")

    predict_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda, main_program=predict_prog)

    test_data_generator = processor.data_generator(
        batch_size=args.batch_size, 
        phase='test',
        epoch=1,
        shuffle=False)
    predict_pyreader.decorate_tensor_provider(test_data_generator)

    predict_pyreader.start()
    all_results = []
    time_begin = time.time()
    while True:
        try:
            results = predict_exe.run(fetch_list=[probs.name])
            all_results.extend(results[0])
        except fluid.core.EOFException:
            predict_pyreader.reset()
            break
    time_end = time.time()

    np.set_printoptions(precision=4, suppress=True)
    print("-------------- prediction results --------------")
    print("example_id\t" + '  '.join(processor.get_labels()))
    if in_tokens[task_name]: 
        for index, result in enumerate(all_results): 
            tags = pred_func(result)
            print("%s\t%s" % (index, tags))
    else: 
        tags = pred_func(all_results, args.max_seq_len)
        for index, tag in enumerate(tags): 
            print("%s\t%s" % (index, tag))
    
    if args.save_inference_model_path:
        _, ckpt_dir = os.path.split(args.init_checkpoint)
        dir_name = ckpt_dir + '_inference_model'
        model_path = os.path.join(args.save_inference_model_path, dir_name)
        fluid.io.save_inference_model(
            model_path,
            feed_target_names, [probs],
            exe,
            main_program=predict_prog)


if __name__ == '__main__': 
    args = parser.parse_args()
    print_arguments(args)
    main(args)
