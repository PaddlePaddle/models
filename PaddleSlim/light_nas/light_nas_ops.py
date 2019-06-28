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
from light_nas_space import LightNASSpace
from get_ops_from_program import get_ops_from_program, write_lookup_table

# get all ops in the search space
space = LightNASSpace()
all_ops = space.get_all_ops(True, True)
write_lookup_table(all_ops, 'lightnas_ops.txt')

# get all ops from mobilenetv2
startup_program, main_program, _, _, _, _, _ = space.create_net()
all_ops = get_ops_from_program(main_program)
all_ops = list(set(all_ops))
write_lookup_table(all_ops, 'mobilenetv2_ops.txt')
