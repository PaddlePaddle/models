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


def cdssm_base():
    """
    set configs
    """
    config = basic_config.config()
    config.learning_rate = 0.001
    config.save_dirname = "model_dir"
    config.use_pretrained_word_embedding = True
    config.dict_dim = 40000  # approx_vocab_size

    # net config
    config.emb_dim = 300
    config.kernel_size = 5
    config.kernel_count = 300
    config.fc_dim = 128
    config.mlp_hid_dim = [128, 128]
    config.droprate_conv = 0.1
    config.droprate_fc = 0.1
    config.class_dim = 2

    return config
