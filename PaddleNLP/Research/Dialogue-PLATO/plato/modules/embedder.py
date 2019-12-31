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
Embedder class.
"""

import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding
from paddle.fluid.dygraph import Layer
import paddle.fluid.layers as layers

import plato.modules.functions as F


class Embedder(Layer):
    """
    Composite embedding layer.
    """

    def __init__(self,
                 name_scope,
                 hidden_dim,
                 num_token_embeddings,
                 num_pos_embeddings,
                 num_type_embeddings,
                 num_turn_embeddings,
                 padding_idx=None,
                 dropout=0.1,
                 pos_trainable=False):
        super().__init__(name_scope)

        self.token_embedding = Embedding(name_scope=self.full_name(),
                                         size=[num_token_embeddings, hidden_dim])
        self.pos_embedding = Embedding(name_scope=self.full_name(),
                                       size=[num_pos_embeddings, hidden_dim],
                                       param_attr=fluid.ParamAttr(trainable=pos_trainable))
        self.type_embedding = Embedding(name_scope=self.full_name(),
                                        size=[num_type_embeddings, hidden_dim])
        self.turn_embedding = Embedding(name_scope=self.full_name(),
                                        size=[num_turn_embeddings, hidden_dim])
        self.dropout = dropout
        return

    def forward(self, token_inp, pos_inp, type_inp, turn_inp):
        embed = self.token_embedding(token_inp) + \
            self.pos_embedding(pos_inp) + \
            self.type_embedding(type_inp) + \
            self.turn_embedding(turn_inp)
        embed = F.dropout(embed, self.dropout)
        return embed


def main():
    import numpy as np

    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = Embedder("Embedder", 10, 20, 20, 20, 20)
        token_inp = fluid.dygraph.to_variable(np.random.randint(0, 19, [10, 10]).astype("int64"))
        pos_inp = fluid.dygraph.to_variable(np.random.randint(0, 19, [10, 10]).astype("int64"))
        type_inp = fluid.dygraph.to_variable(np.random.randint(0, 19, [10, 10]).astype("int64"))
        turn_inp = fluid.dygraph.to_variable(np.random.randint(0, 19, [10, 10]).astype("int64"))
        out = model(token_inp, pos_inp, type_inp, turn_inp)
        print(out)


if __name__ == "__main__":
    main()
