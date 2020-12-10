# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from pgl.contrib.imperative.message_passing import SageConv


class ErnieSageV2Conv(SageConv):
    """ ErnieSage (abbreviation of ERNIE SAmple aggreGatE), a model proposed by the PGL team.
    ErnieSageV2: Ernie is applied to the EDGE of the text graph.
    """

    def __init__(self, ernie, input_size, hidden_size, initializer,
                 learning_rate, agg, name):
        """ErnieSageV2: Ernie is applied to the EDGE of the text graph.

        Args:
            ernie (nn.Layer): the ernie model.
            input_size (int): input size of feature tensor.
            hidden_size (int): hidden size of the Conv layers.
            initializer (initializer): parameters initializer.
            learning_rate (float): learning rate.
            agg (str): aggregate function. 'sum', 'mean', 'max' avaliable.
            name (str): layer name.
        """
        super(ErnieSageV2Conv, self).__init__(
            input_size, hidden_size, initializer, learning_rate, "sum", name)
        self.ernie = ernie

    def ernie_send(self, src_feat, dst_feat, edge_feat):
        """ Apply ernie model on the edge.

        Args:
            src_feat (Tensor Dict): src feature tensor dict.
            dst_feat (Tensor Dict): dst feature tensor dict.
            edge_feat (Tensor Dict): edge feature tensor dict.

        Returns:
            Tensor Dict: tensor dict which use 'msg' as the key.
        """
        # input_ids
        cls = paddle.full(
            shape=[src_feat["term_ids"].shape[0], 1],
            dtype="int64",
            fill_value=1)
        src_ids = paddle.concat([cls, src_feat["term_ids"]], 1)
        dst_ids = dst_feat["term_ids"]

        # sent_ids
        sent_ids = paddle.concat(
            [paddle.zeros_like(src_ids), paddle.ones_like(dst_ids)], 1)
        term_ids = paddle.concat([src_ids, dst_ids], 1)

        # build position_ids
        input_mask = paddle.cast(term_ids > 0, "int64")
        position_ids = paddle.cumsum(input_mask, axis=1) - 1

        outputs = self.ernie(term_ids, sent_ids, position_ids)
        feature = outputs[1]
        return {"msg": feature}

    def send_recv(self, graph, feature):
        """Message Passing of erniesage v2.

        Args:
            graph (GraphTensor): the GraphTensor object.
            feature (Tensor): the node feature tensor.

        Returns:
            Tensor: the self and neighbor feature tensors.
        """
        msg = graph.send(self.ernie_send, nfeat_list=[("term_ids", feature)])
        neigh_feature = graph.recv(msg, self.agg_func)

        term_ids = feature
        cls = paddle.full(
            shape=[term_ids.shape[0], 1], dtype="int64", fill_value=1)
        term_ids = paddle.concat([cls, term_ids], 1)
        term_ids.stop_gradient = True
        outputs = self.ernie(term_ids, paddle.zeros_like(term_ids))
        self_feature = outputs[1]

        return self_feature, neigh_feature
