# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import copy
from paddle.io import DataLoader

__all__ = ["GraphDataLoader"]


class GraphDataLoader(object):
    def __init__(self, dataset):
        self.loader = DataLoader(dataset)
        self.graphs = dataset.graphs

    def __iter__(self):
        func = self.__callback__()
        for data in self.loader():
            yield func(data)

    def __call__(self):
        return self.__iter__()

    def __callback__(self):
        """ callback function, for recontruct a dict or graph.
        """

        def construct(tensors):
            """ tensor list to ([graph_tensor, graph_tensor, ...], 
            other tensor) 
            """
            start_len = 0
            datas = []
            graph_list = []
            for i in range(len(tensors)):
                tensors[i] = paddle.squeeze(tensors[i], axis=0)

            for graph in self.graphs:
                new_graph = copy.deepcopy(graph)
                length = len(new_graph._graph_attr_holder)
                graph_tensor_list = tensors[start_len:start_len + length]
                start_len += length
                graph_list.append(new_graph.from_tensor(graph_tensor_list))

            for i in range(start_len, len(tensors)):
                datas.append(tensors[i])
            return graph_list, datas

        return construct
