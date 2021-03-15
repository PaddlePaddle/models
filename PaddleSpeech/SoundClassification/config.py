
# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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

sample_rate=32000
window_size=1024
hop_size=320
mel_bins=64
fmin=50
fmax=14000
audioset_checkpoint='./assets/Cnn14_class=527mAP=0.431.pd.tar'
batch_size=16
num_class=50#esc50
mel_sample_len = 256+128#+64
epoch_num = 200
dropout=0.5
esc50_audio = './data/ESC-50-master/audio'
esc50_mel = './data/ESC-50-master/mel'
model_path = './esc50_checkpoints/'
log_path = './log'
