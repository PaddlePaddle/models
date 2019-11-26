#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Convert Google n-gram mask reading comprehension models to Fluid parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import collections
from args import print_arguments
import tensorflow as tf
import paddle.fluid as fluid
from tensorflow.python import pywrap_tensorflow


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--init_tf_checkpoint",
        type=str,
        required=True,
        help="Initial TF checkpoint (a pre-trained BERT model).")

    parser.add_argument(
        "--fluid_params_dir",
        type=str,
        required=True,
        help="The directory to store converted Fluid parameters.")
    args = parser.parse_args()
    return args


def parse(init_checkpoint):
    tf_fluid_param_name_map = collections.OrderedDict()
    tf_param_name_shape_map = collections.OrderedDict()

    init_vars = tf.train.list_variables(init_checkpoint)
    for (var_name, var_shape) in init_vars: 
        print("%s\t%s" % (var_name, var_shape))
        fluid_param_name = ''
        if var_name.startswith('bert/'):
            key = var_name[5:]
            if (key.startswith('embeddings/')):
                if (key.endswith('LayerNorm/gamma')):
                    fluid_param_name = 'pre_encoder_layer_norm_scale'
                elif (key.endswith('LayerNorm/beta')):
                    fluid_param_name = 'pre_encoder_layer_norm_bias'
                elif (key.endswith('position_embeddings')):
                    fluid_param_name = 'pos_embedding'
                elif (key.endswith('word_embeddings')):
                    fluid_param_name = 'word_embedding'
                elif (key.endswith('token_type_embeddings')):
                    fluid_param_name = 'sent_embedding'
                else:
                    print("ignored param: %s" % var_name)
            elif (key.startswith('encoder/')):
                key = key[8:]
                layer_num = int(key[key.find('_') + 1:key.find('/')])
                suffix = "encoder_layer_" + str(layer_num)
                if key.endswith('attention/output/LayerNorm/beta'):
                    fluid_param_name = suffix + '_post_att_layer_norm_bias'
                elif key.endswith('attention/output/LayerNorm/gamma'):
                    fluid_param_name = suffix + '_post_att_layer_norm_scale'
                elif key.endswith('attention/output/dense/bias'):
                    fluid_param_name = suffix + '_multi_head_att_output_fc.b_0'
                elif key.endswith('attention/output/dense/kernel'):
                    fluid_param_name = suffix + '_multi_head_att_output_fc.w_0'
                elif key.endswith('attention/self/key/bias'):
                    fluid_param_name = suffix + '_multi_head_att_key_fc.b_0'
                elif key.endswith('attention/self/key/kernel'):
                    fluid_param_name = suffix + '_multi_head_att_key_fc.w_0'
                elif key.endswith('attention/self/query/bias'):
                    fluid_param_name = suffix + '_multi_head_att_query_fc.b_0'
                elif key.endswith('attention/self/query/kernel'):
                    fluid_param_name = suffix + '_multi_head_att_query_fc.w_0'
                elif key.endswith('attention/self/value/bias'):
                    fluid_param_name = suffix + '_multi_head_att_value_fc.b_0'
                elif key.endswith('attention/self/value/kernel'):
                    fluid_param_name = suffix + '_multi_head_att_value_fc.w_0'
                elif key.endswith('intermediate/dense/bias'):
                    fluid_param_name = suffix + '_ffn_fc_0.b_0'
                elif key.endswith('intermediate/dense/kernel'):
                    fluid_param_name = suffix + '_ffn_fc_0.w_0'
                elif key.endswith('output/LayerNorm/beta'):
                    fluid_param_name = suffix + '_post_ffn_layer_norm_bias'
                elif key.endswith('output/LayerNorm/gamma'):
                    fluid_param_name = suffix + '_post_ffn_layer_norm_scale'
                elif key.endswith('output/dense/bias'):
                    fluid_param_name = suffix + '_ffn_fc_1.b_0'
                elif key.endswith('output/dense/kernel'):
                    fluid_param_name = suffix + '_ffn_fc_1.w_0'
                else:
                    print("ignored param: %s" % var_name)
            elif (key.startswith('pooler/')):
                if key.endswith('dense/bias'):
                    fluid_param_name = 'pooled_fc.b_0'
                elif key.endswith('dense/kernel'):
                    fluid_param_name = 'pooled_fc.w_0'
                else:
                    print("ignored param: %s" % var_name)
            else:
                print("ignored param: %s" % var_name)

        elif var_name.startswith('output/'):
            if var_name == 'output/passage_regression/weights':
                fluid_param_name = 'passage_regression_weights'
            elif var_name == 'output/span/start/weights':
                fluid_param_name = 'span_start_weights'
            elif var_name == "output/span/end/conditional/dense/kernel": 
                fluid_param_name = 'conditional_fc_weights'
            elif var_name == "output/span/end/conditional/dense/bias": 
                fluid_param_name = 'conditional_fc_bias'
            elif var_name == "output/span/end/conditional/LayerNorm/beta": 
                fluid_param_name = 'conditional_layernorm_beta'
            elif var_name == "output/span/end/conditional/LayerNorm/gamma": 
                fluid_param_name = 'conditional_layernorm_gamma'
            elif var_name == "output/span/end/weights": 
                fluid_param_name = 'span_end_weights'
            else:
                print("ignored param: %s" % var_name)
        else:
            print("ignored param: %s" % var_name)

        if fluid_param_name != '':
            tf_fluid_param_name_map[var_name] = fluid_param_name
            tf_param_name_shape_map[var_name] = var_shape
            fluid_param_name = ''

    return tf_fluid_param_name_map, tf_param_name_shape_map


def convert(args):
    tf_fluid_param_name_map, tf_param_name_shape_map = parse(
        args.init_tf_checkpoint)
    program = fluid.Program()
    global_block = program.global_block()
    for param in tf_fluid_param_name_map:
        global_block.create_parameter(
            name=tf_fluid_param_name_map[param],
            shape=tf_param_name_shape_map[param],
            dtype='float32',
            initializer=fluid.initializer.Constant(value=0.0))

    place = fluid.core.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(program)

    print('---------------------- Converted Parameters -----------------------')
    print('###### [TF param name] --> [Fluid param name]  [param shape] ######')
    print('-------------------------------------------------------------------')

    reader = pywrap_tensorflow.NewCheckpointReader(args.init_tf_checkpoint)
    for param in tf_fluid_param_name_map:
        value = reader.get_tensor(param)
        fluid.global_scope().find_var(tf_fluid_param_name_map[
            param]).get_tensor().set(value, place)
        print(param, ' --> ', tf_fluid_param_name_map[param], '  ', value.shape)

    fluid.io.save_params(exe, args.fluid_params_dir, main_program=program)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    convert(args)
