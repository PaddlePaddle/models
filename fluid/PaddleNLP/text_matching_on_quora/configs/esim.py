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

import basic_config

def esim_seq():
    """
    set configs
    """
    config = basic_config.config()
    config.optimizer_type = 'adam'
    config.learning_rate = 0.0004
    config.save_dirname = "model_dir"
    config.use_pretrained_word_embedding = True
    config.dict_dim = 40000 # approx_vocab_size
    config.batch_size = 128
    config.use_cuda = True

    # net config
    config.emb_dim = 300
    config.lstm_hid_dim = 300
    config.mlp_hid_dim = 300
    config.class_dim = 2
    config.drop_rate = 0.5
    return config

