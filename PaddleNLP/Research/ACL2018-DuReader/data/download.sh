#!/bin/bash
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# download preprocessed data
wget -c --no-check-certificate https://baidu-nlp.bj.bcebos.com/dureader_machine_reading-dataset-2.0.0.tar.gz
# download trained model parameters and vocabulary
wget -c --no-check-certificate https://baidu-nlp.bj.bcebos.com/dureader_machine_reading-bidaf-1.0.0.tar.gz 

# decompression
tar -zxvf dureader_machine_reading-dataset-2.0.0.tar.gz
tar -zxvf dureader_machine_reading-bidaf-1.0.0.tar.gz 

ln -s trained_model_para/vocab ./
ln -s trained_model_para/saved_model ./

