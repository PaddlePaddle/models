# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle
import sys
sys.path.append('..')
from light_nas_space import get_tokens_set, LightNASSpace
from get_ops_from_program import get_ops_from_program, write_lookup_table

space = LightNASSpace()
tokens = get_tokens_set()

n = len(tokens)
all_ops = []
for idx in range(0, n):
    current_token = tokens[idx]
    startup_program, main_program, _, _, _, _, _ = space.create_net(current_token)
    op_params = get_ops_from_program(main_program)
    all_ops = all_ops + op_params
    all_ops = list(set(all_ops))
    if (idx+1)%10 == 0:
        write_lookup_table(all_ops, 'lightnas_ops_tmp.txt')
        print('current file number is: {}'.format(idx))
        print('current number of ops is:', len(all_ops))

write_lookup_table(all_ops, 'lightnas_ops.txt')
print('{} networks have {} ops in total'.format(n, len(all_ops)))
