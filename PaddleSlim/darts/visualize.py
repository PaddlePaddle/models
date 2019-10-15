# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import functools
import utility
import genotypes
from graphviz import Digraph

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utility.add_arguments, argparser=parser)

add_arg('epoch', int, 0, "Epoch.")
add_arg('loss', float, 0.001, "Loss.")
add_arg('acc', float, 0.999, "Acc.")

args = parser.parse_args()


def plot(genotype_normal, genotype_reduce, filename):
    g = Digraph(
        format='png',
        edge_attr=dict(fontname="times"),
        node_attr=dict(
            style='filled',
            shape='ellipse',
            align='center',
            height='0.5',
            width='0.5',
            penwidth='2',
            fontname="times"),
        engine='dot')

    g.body.extend(['rankdir=LR'])
    g.attr(label='<Epoch {}, Loss:{},<font color="red"><b> Acc:{}</b></font>>'.
           format(args.epoch, args.loss, args.acc))

    g.node("n_c_{k-2}", fillcolor='darkseagreen2')
    g.node("n_c_{k-1}", fillcolor='darkseagreen2')
    g.node("r_c_{k-2}", fillcolor='darkseagreen2')
    g.node("r_c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype_normal) % 2 == 0
    steps = len(genotype_normal) // 2

    for i in range(steps):
        g.node('n_' + str(i), fillcolor='lightblue')
    for i in range(steps):
        g.node('r_' + str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype_normal[k]
            if j == 0:
                u = "n_c_{k-2}"
            elif j == 1:
                u = "n_c_{k-1}"
            else:
                u = 'n_' + str(j - 2)
            v = 'n_' + str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype_reduce[k]
            if j == 0:
                u = "r_c_{k-2}"
            elif j == 1:
                u = "r_c_{k-1}"
            else:
                u = 'r_' + str(j - 2)
            v = 'r_' + str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("n_c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge('n_' + str(i), "n_c_{k}", fillcolor="gray")
    g.node("r_c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge('r_' + str(i), "r_c_{k}", fillcolor="gray")
    g.render(filename, view=False)


if __name__ == '__main__':
    genotype_name = 'pass' + str(args.epoch)
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    prefix = 'search_0' if args.epoch <= 9 else 'search_'
    plot(genotype.normal, genotype.reduce, prefix + str(args.epoch))
