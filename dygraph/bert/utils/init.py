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

from __future__ import print_function

import os
import six
import ast
import copy

import numpy as np
import paddle.fluid as fluid


def cast_fp32_to_fp16(exe, main_program):
    print("Cast parameters to float16 data format.")
    for param in main_program.global_block().all_parameters():
        if not param.name.endswith(".master"):
            param_t = fluid.global_scope().find_var(param.name).get_tensor()
            data = np.array(param_t)
            if param.name.find("layer_norm") == -1:
                param_t.set(np.float16(data).view(np.uint16), exe.place)
            master_param_var = fluid.global_scope().find_var(param.name +
                                                             ".master")
            if master_param_var is not None:
                master_param_var.get_tensor().set(data, exe.place)


def init_checkpoint(exe, init_checkpoint_path, main_program, use_fp16=False):
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print("Load model from {}".format(init_checkpoint_path))

    if use_fp16:
        cast_fp32_to_fp16(exe, main_program)


def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program,
                            use_fp16=False):
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params)
    print("Load pretraining parameters from {}.".format(
        pretraining_params_path))

    if use_fp16:
        cast_fp32_to_fp16(exe, main_program)


def init_from_static_model(dir_path, cls_model, bert_config):
    def load_numpy_weight(file_name):
        if six.PY2:
            res = np.load(os.path.join(dir_path, file_name), allow_pickle=True)
        else:
            res = np.load(
                os.path.join(dir_path, file_name),
                allow_pickle=True,
                encoding='latin1')
        assert res is not None
        return res

    # load word embedding
    _param = load_numpy_weight("word_embedding")
    cls_model.bert_layer._src_emb.set_dict({"weight": _param})
    print("INIT word embedding")

    _param = load_numpy_weight("pos_embedding")
    cls_model.bert_layer._pos_emb.set_dict({"weight": _param})
    print("INIT pos embedding")

    _param = load_numpy_weight("sent_embedding")
    cls_model.bert_layer._sent_emb.set_dict({"weight": _param})
    print("INIT sent embedding")

    _param0 = load_numpy_weight("pooled_fc.w_0")
    _param1 = load_numpy_weight("pooled_fc.b_0")
    cls_model.bert_layer.pooled_fc.set_dict({
        "weight": _param0,
        "bias": _param1
    })
    print("INIT pooled_fc")

    _param0 = load_numpy_weight("pre_encoder_layer_norm_scale")
    _param1 = load_numpy_weight("pre_encoder_layer_norm_bias")
    cls_model.bert_layer.pre_process_layer._sub_layers["layer_norm_0"].set_dict(
        {
            "weight": _param0,
            "bias": _param1
        })
    print("INIT pre_encoder layer norm")

    for _i in range(bert_config["num_hidden_layers"]):
        _param_weight = "encoder_layer_%d_multi_head_att_query_fc.w_0" % _i
        _param_bias = "encoder_layer_%d_multi_head_att_query_fc.b_0" % _i

        _param_weight = load_numpy_weight(_param_weight)
        _param_bias = load_numpy_weight(_param_bias)

        cls_model.bert_layer._encoder._sub_layers[
            "esl_%d" % _i]._multihead_attention_layer._q_fc.set_dict({
                "weight": _param_weight,
                "bias": _param_bias
            })
        print("INIT multi_head_att_query_fc %d" % _i)

        _param_weight = "encoder_layer_%d_multi_head_att_key_fc.w_0" % _i
        _param_bias = "encoder_layer_%d_multi_head_att_key_fc.b_0" % _i

        _param_weight = load_numpy_weight(_param_weight)
        _param_bias = load_numpy_weight(_param_bias)

        cls_model.bert_layer._encoder._sub_layers[
            "esl_%d" % _i]._multihead_attention_layer._k_fc.set_dict({
                "weight": _param_weight,
                "bias": _param_bias
            })
        print("INIT multi_head_att_key_fc %d" % _i)

        _param_weight = "encoder_layer_%d_multi_head_att_value_fc.w_0" % _i
        _param_bias = "encoder_layer_%d_multi_head_att_value_fc.b_0" % _i

        _param_weight = load_numpy_weight(_param_weight)
        _param_bias = load_numpy_weight(_param_bias)

        cls_model.bert_layer._encoder._sub_layers[
            "esl_%d" % _i]._multihead_attention_layer._v_fc.set_dict({
                "weight": _param_weight,
                "bias": _param_bias
            })
        print("INIT multi_head_att_value_fc %d" % _i)

        # init output fc
        _param_weight = "encoder_layer_%d_multi_head_att_output_fc.w_0" % _i
        _param_bias = "encoder_layer_%d_multi_head_att_output_fc.b_0" % _i

        _param_weight = load_numpy_weight(_param_weight)
        _param_bias = load_numpy_weight(_param_bias)

        cls_model.bert_layer._encoder._sub_layers[
            "esl_%d" % _i]._multihead_attention_layer._proj_fc.set_dict({
                "weight": _param_weight,
                "bias": _param_bias
            })
        print("INIT multi_head_att_output_fc %d" % _i)

        # init layer_norm 1
        _param_weight = "encoder_layer_%d_post_att_layer_norm_scale" % _i
        _param_bias = "encoder_layer_%d_post_att_layer_norm_bias" % _i

        _param_weight = load_numpy_weight(_param_weight)
        _param_bias = load_numpy_weight(_param_bias)

        cls_model.bert_layer._encoder._sub_layers[
            "esl_%d" % _i]._postprocess_layer.layer_norm_0.set_dict({
                "weight": _param_weight,
                "bias": _param_bias
            })
        print("INIT layer norm in attention at %d layer" % _i)

        # init layer_norm 2
        _param_weight = "encoder_layer_%d_post_ffn_layer_norm_scale" % _i
        _param_bias = "encoder_layer_%d_post_ffn_layer_norm_bias" % _i

        _param_weight = load_numpy_weight(_param_weight)
        _param_bias = load_numpy_weight(_param_bias)

        cls_model.bert_layer._encoder._sub_layers[
            "esl_%d" % _i]._postprocess_layer2.layer_norm_0.set_dict({
                "weight": _param_weight,
                "bias": _param_bias
            })
        print("INIT layer norm in FFN at %d layer" % _i)

        # init FFN 1
        _param_weight = "encoder_layer_%d_ffn_fc_0.w_0" % _i
        _param_bias = "encoder_layer_%d_ffn_fc_0.b_0" % _i

        _param_weight = load_numpy_weight(_param_weight)
        _param_bias = load_numpy_weight(_param_bias)

        cls_model.bert_layer._encoder._sub_layers[
            "esl_%d" % _i]._positionwise_feed_forward._i2h.set_dict({
                "weight": _param_weight,
                "bias": _param_bias
            })
        print("INIT FFN-1 at %d layer" % _i)

        # init FFN 2
        _param_weight = "encoder_layer_%d_ffn_fc_1.w_0" % _i
        _param_bias = "encoder_layer_%d_ffn_fc_1.b_0" % _i

        _param_weight = load_numpy_weight(_param_weight)
        _param_bias = load_numpy_weight(_param_bias)

        cls_model.bert_layer._encoder._sub_layers[
            "esl_%d" % _i]._positionwise_feed_forward._h2o.set_dict({
                "weight": _param_weight,
                "bias": _param_bias
            })
        print("INIT FFN-2 at %d layer" % _i)

    # init cls fc
    #_param_weight = "cls_out_w"
    #_param_bias = "cls_out_b"

    #_param_weight = load_numpy_weight(_param_weight)
    #_param_bias = load_numpy_weight(_param_bias)

    #cls_model.cls_fc.set_dict({"weight":_param_weight, "bias":_param_bias})
    #print("INIT CLS FC layer")
    return True
