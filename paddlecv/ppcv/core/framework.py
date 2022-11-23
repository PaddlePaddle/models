# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
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

import os
import numpy as np
import math
import paddle
from collections import defaultdict
import ppcv
from ppcv.ops import *
from ppcv.utils.helper import get_output_keys, gen_input_name
from ppcv.core.workspace import create


class DAG(object):
    """
    Directed Acyclic Graph(DAG) engine, builds one DAG topology.
    """

    def __init__(self, cfg):
        self.graph, self.rev_graph, self.in_degrees = self.build_dag(cfg)
        self.num = len(self.in_degrees)

    def build_dag(self, cfg):
        graph = defaultdict(list)  # op -> next_op
        unique_name = set()
        unique_name.add('input')
        rev_graph = defaultdict(list)  # op -> last_op
        for op in cfg:
            op_dict = list(op.values())[0]
            unique_name.add(op_dict['name'])

        in_degrees = dict((u, 0) for u in unique_name)
        for op in cfg:
            op_cfg = list(op.values())[0]
            inputs = op_cfg['Inputs']
            for input in inputs:
                last_op = input.split('.')[0]
                graph[last_op].append(op_cfg['name'])
                rev_graph[op_cfg['name']].append(last_op)
                in_degrees[op_cfg['name']] += 1
        return graph, rev_graph, in_degrees

    def get_graph(self):
        return self.graph

    def get_reverse_graph(self):
        return self.rev_graph

    def topo_sort(self):
        """
        Topological sort of DAG, creates inverted multi-layers views.
        Args:
            graph (dict): the DAG stucture
            in_degrees (dict): Next op list for each op
        Returns:
            sort_result: the hierarchical topology list. examples:
                DAG :[A -> B -> C -> E]
                            \-> D /
                sort_result: [A, B, C, D, E]
        """

        # Select vertices with in_degree = 0
        Q = [u for u in self.in_degrees if self.in_degrees[u] == 0]
        sort_result = []
        while Q:
            u = Q.pop()
            sort_result.append(u)
            for v in self.graph[u]:
                # remove output degrees
                self.in_degrees[v] -= 1
                # re-select vertices with in_degree = 0
                if self.in_degrees[v] == 0:
                    Q.append(v)
        if len(sort_result) == self.num:
            return sort_result
        else:
            return None


class Executor(object):
    """
    The executor which implements model series pipeline

    Args:
        model_cfg: The models configuration
        env_cfg: The enrionment configuration
    """

    def __init__(self, model_cfg, env_cfg):
        dag = DAG(model_cfg)
        self.order = dag.topo_sort()
        self.model_cfg = model_cfg

        self.op_name2op = {}
        self.has_output_op = False
        for op in model_cfg:
            op_arch = list(op.keys())[0]
            op_cfg = list(op.values())[0]
            op_name = op_cfg['name']
            op = create(op_arch, op_cfg, env_cfg)
            self.op_name2op[op_name] = op
            if op.type() == 'OUTPUT':
                self.has_output_op = True

        self.output_keys = get_output_keys(model_cfg)
        self.last_ops_dict = dag.get_reverse_graph()
        self.input_dep = self.reset_dep()

    def reset_dep(self, ):
        return self.build_dep(self.model_cfg, self.output_keys)

    def build_dep(self, cfg, output_keys):
        # compute the output degree for each input name
        dep = dict()
        for op in cfg:
            inputs = list(op.values())[0]['Inputs']
            for name in inputs:
                if name in dep:
                    dep[name] += 1
                else:
                    dep.update({name: 1})
        return dep

    def update_res(self, results, op_outputs, input_name):
        # step1: remove the result when keys not used in later input
        for res, out in zip(results, op_outputs):
            if self.has_output_op:
                del_name = []
                for k in out.keys():
                    if k not in self.input_dep:
                        del_name.append(k)
                # remove the result when keys not used in later input
                for name in del_name:
                    del out[name]
            res.update(out)

        # step2: if the input name is no longer used, then result will be deleted  
        if self.has_output_op:
            for name in input_name:
                self.input_dep[name] -= 1
                if self.input_dep[name] == 0:
                    for res in results:
                        del res[name]

    def run(self, input, frame_id=-1):
        self.input_dep = self.reset_dep()
        # execute each operator according to toposort order
        results = input
        for i, op_name in enumerate(self.order[1:]):
            op = self.op_name2op[op_name]
            op.set_frame(frame_id)
            last_ops = self.last_ops_dict[op_name]
            input_keys = op.get_input_keys()
            output_keys = list(results[0].keys())
            input = op.filter_input(results, input_keys)
            last_op_output = op(input)
            if op.type() != 'OUTPUT':
                op.check_output(last_op_output, op_name)
                self.update_res(results, last_op_output, input_keys)
            else:
                results = last_op_output

        return results
