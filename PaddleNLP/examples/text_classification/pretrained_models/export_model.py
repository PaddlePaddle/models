# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlenlp.data import Stack, Tuple, Pad
import paddlenlp as ppnlp

MODEL_CLASSES = {
    "bert": (ppnlp.transformers.BertForSequenceClassification,
             ppnlp.transformers.BertTokenizer),
    'ernie': (ppnlp.transformers.ErnieForSequenceClassification,
              ppnlp.transformers.ErnieTokenizer),
    'roberta': (ppnlp.transformers.RobertaForSequenceClassification,
                ppnlp.transformers.RobertaTokenizer),
}


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_type", default='roberta', required=True, type=str, help="Model type selected in the list: " +", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='roberta-wwm-ext', required=True, type=str, help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(sum([list(classes[-1].pretrained_init_configuration.keys()) for classes in MODEL_CLASSES.values()], [])))
    parser.add_argument("--params_path", type=str, required=True, default='./checkpoint/model_200/model_state.pdparams', help="The path to model parameters to be loaded.")
    parser.add_argument("--output_path", type=str, default='./static_graph_params', help="The path of model parameter in static graph to be saved.")
    args = parser.parse_args()
    return args
# yapf: enable

if __name__ == "__main__":
    args = parse_args()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.model_name_or_path == 'ernie-tiny':
        # ErnieTinyTokenizer is special for ernie-tiny pretained model.
        tokenizer = ppnlp.transformers.ErnieTinyTokenizer.from_pretrained(
            args.model_name_or_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    data = [
        '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般',
        '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
        '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
    ]
    label_map = {0: 'negative', 1: 'positive'}

    model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=len(label_map))

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    model.eval()

    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")  # segment_ids
        ])
    # Save in static graph model.
    paddle.jit.save(model, args.output_path)
