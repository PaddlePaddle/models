#!/usr/bin/env bash
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

# download pretrained ERNIE model
wget --no-check-certificate https://ernie.bj.bcebos.com/ERNIE_en_1.0.tgz
tar -xvf ERNIE_en_1.0.tgz
rm ERNIE_en_1.0.tgz
ln -s ERNIE_en_1.0 ernie_model
