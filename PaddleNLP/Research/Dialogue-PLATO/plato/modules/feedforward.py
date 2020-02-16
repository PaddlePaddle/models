#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
FeedForward class.
"""

import paddle.fluid as fluid
from paddle.fluid.dygraph import FC
from paddle.fluid.dygraph import Layer
import paddle.fluid.layers as layers

import plato.modules.functions as F


class FeedForward(Layer):
    """
    Positional feed forward layer.
    """

    def __init__(self, name_scope, hidden_dim, inner_dim, dropout):
        super().__init__(name_scope)

        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.linear_hidden = FC(name_scope=self.full_name(),
                                size=inner_dim,
                                num_flatten_dims=2,
                                act="gelu")
        self.linear_out = FC(name_scope=self.full_name(),
                             size=hidden_dim,
                             num_flatten_dims=2)
        self.dropout = dropout
        return

    def forward(self, x):
        out = self.linear_hidden(x)
        out = F.dropout(out, self.dropout)
        out = self.linear_out(out)
        return out


def main():
    import numpy as np

    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = FeedForward("FeedForward", 10, 20, 0.5)
        inp = np.random.rand(2, 3, 10).astype("float32")
        inp = fluid.dygraph.to_variable(inp)
        out = model(inp)
        print(out)


if __name__ == "__main__":
    main()
