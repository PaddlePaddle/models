# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from . import basic_config


def infer_sent_v1():
    """
    set configs
    """
    config = basic_config.config()
    config.learning_rate = 0.1
    config.lr_decay = 0.99
    config.optimizer_type = 'sgd'
    config.save_dirname = "model_dir"
    config.use_pretrained_word_embedding = True
    config.dict_dim = 40000  # approx_vocab_size
    config.class_dim = 2

    # net config
    config.emb_dim = 300
    config.droprate_lstm = 0.0
    config.droprate_fc = 0.0
    config.word_embedding_trainable = False
    config.rnn_hid_dim = 2048
    config.mlp_non_linear = False

    return config


def infer_sent_v2():
    """
    use our own config
    """
    config = basic_config.config()
    config.learning_rate = 0.0002
    config.lr_decay = 0.99
    config.optimizer_type = 'adam'
    config.save_dirname = "model_dir"
    config.use_pretrained_word_embedding = True
    config.dict_dim = 40000  # approx_vocab_size
    config.class_dim = 2

    # net config
    config.emb_dim = 300
    config.droprate_lstm = 0.0
    config.droprate_fc = 0.2
    config.word_embedding_trainable = False
    config.rnn_hid_dim = 2048
    config.mlp_non_linear = True

    return config
