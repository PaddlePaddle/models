#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA

from ppdet.core.workspace import register

__all__ = ['MultiBoxHead']


@register
class MultiBoxHead(object):
    __op__ = fluid.layers.multi_box_head
    __append_doc__ = True

    def __init__(self,
                 min_ratio=20,
                 max_ratio=90,
                 min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
                 max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
                 aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.], [2., 3.]],
                 base_size=300,
                 offset=0.5,
                 flip=True):
        super(MultiBoxHead, self).__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios
        self.base_size = base_size
        self.offset = offset
        self.flip = flip
