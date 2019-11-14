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


def decatt_glove():
    """
    use config 'decAtt_glove' in the paper 'Neural Paraphrase Identification of Questions with Noisy Pretraining'
    """
    config = basic_config.config()
    config.learning_rate = 0.05
    config.save_dirname = "model_dir"
    config.use_pretrained_word_embedding = True
    config.dict_dim = 40000  # approx_vocab_size
    config.metric_type = ['accuracy', 'accuracy_with_threshold']
    config.optimizer_type = 'sgd'
    config.lr_decay = 1
    config.use_lod_tensor = False
    config.embedding_norm = False
    config.OOV_fill = 'uniform'
    config.duplicate_data = False

    # net config
    config.emb_dim = 300
    config.proj_emb_dim = 200  #TODO: has project?
    config.num_units = [400, 200]
    config.word_embedding_trainable = True
    config.droprate = 0.1
    config.share_wight_btw_seq = True
    config.class_dim = 2

    return config


def decatt_word():
    """
    use config 'decAtt_glove' in the paper 'Neural Paraphrase Identification of Questions with Noisy Pretraining'
    """
    config = basic_config.config()
    config.learning_rate = 0.05
    config.save_dirname = "model_dir"
    config.use_pretrained_word_embedding = False
    config.dict_dim = 40000  # approx_vocab_size
    config.metric_type = ['accuracy', 'accuracy_with_threshold']
    config.optimizer_type = 'sgd'
    config.lr_decay = 1
    config.use_lod_tensor = False
    config.embedding_norm = False
    config.OOV_fill = 'uniform'
    config.duplicate_data = False

    # net config
    config.emb_dim = 300
    config.proj_emb_dim = 200  #TODO: has project?
    config.num_units = [400, 200]
    config.word_embedding_trainable = True
    config.droprate = 0.1
    config.share_wight_btw_seq = True
    config.class_dim = 2

    return config
