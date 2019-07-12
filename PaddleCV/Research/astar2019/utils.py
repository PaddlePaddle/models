"""Contains common utility functions."""
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from prettytable import PrettyTable
import distutils.util
import numpy as np
import six


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def summary(main_prog):
    '''
    It can summary model's PARAMS, FLOPs until now.
    It support common operator like conv, fc, pool, relu, sigmoid, bn etc.
    Args:
        main_prog: main program
    Returns:
        print summary on terminal
    '''
    collected_ops_list = []
    is_quantize = False
    for one_b in main_prog.blocks:
        block_vars = one_b.vars
        for one_op in one_b.ops:
            if str(one_op.type).find('quantize') > -1:
                is_quantize = True
            op_info = OrderedDict()
            spf_res = _summary_model(block_vars, one_op)
            if spf_res is None:
                continue
            # TODO: get the operator name
            op_info['type'] = one_op.type
            op_info['input_shape'] = spf_res[0][1:]
            op_info['out_shape'] = spf_res[1][1:]
            op_info['PARAMs'] = spf_res[2]
            op_info['FLOPs'] = spf_res[3]
            collected_ops_list.append(op_info)


    summary_table, total = _format_summary(collected_ops_list)
    _print_summary(summary_table, total)
    return total, is_quantize


def _summary_model(block_vars, one_op):
    '''
    Compute operator's params and flops.
    Args:
        block_vars: all vars of one block
        one_op: one operator to count
    Returns:
        in_data_shape: one operator's input data shape
        out_data_shape: one operator's output data shape
        params: one operator's PARAMs
        flops: : one operator's FLOPs
    '''
    if one_op.type in ['conv2d', 'depthwise_conv2d']:
        k_arg_shape = block_vars[one_op.input("Filter")[0]].shape
        in_data_shape = block_vars[one_op.input("Input")[0]].shape
        out_data_shape = block_vars[one_op.output("Output")[0]].shape
        c_out, c_in, k_h, k_w = k_arg_shape
        _, c_out_, h_out, w_out = out_data_shape
        assert c_out == c_out_, 'shape error!'
        k_groups = one_op.attr("groups")
        kernel_ops = k_h * k_w * (c_in / k_groups)
        bias_ops = 0 if one_op.input("Bias") == [] else 1
        params = c_out * (kernel_ops + bias_ops)
        flops = h_out * w_out * c_out * (kernel_ops + bias_ops)
        # base nvidia paper, include mul and add
        flops = 2 * flops

        # var_name = block_vars[one_op.input("Filter")[0]].name
        # if var_name.endswith('.int8'):
        #     flops /= 2.0

    elif one_op.type == 'pool2d':
        in_data_shape = block_vars[one_op.input("X")[0]].shape
        out_data_shape = block_vars[one_op.output("Out")[0]].shape
        _, c_out, h_out, w_out = out_data_shape
        k_size = one_op.attr("ksize")
        params = 0
        flops = h_out * w_out * c_out * (k_size[0] * k_size[1])

    elif one_op.type == 'mul':
        k_arg_shape = block_vars[one_op.input("Y")[0]].shape
        in_data_shape = block_vars[one_op.input("X")[0]].shape
        out_data_shape = block_vars[one_op.output("Out")[0]].shape
        # TODO: fc has mul ops
        # add attr to mul op, tell us whether it belongs to 'fc'
        # this's not the best way
        if 'fc' not in one_op.output("Out")[0]:
            return None
        k_in, k_out = k_arg_shape
        # bias in sum op
        params = k_in * k_out + 1
        flops = k_in * k_out

        # var_name = block_vars[one_op.input("Y")[0]].name
        # if var_name.endswith('.int8'):
        #     flops /= 2.0

    elif one_op.type in ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'prelu']:
        in_data_shape = block_vars[one_op.input("X")[0]].shape
        out_data_shape = block_vars[one_op.output("Out")[0]].shape
        params = 0
        if one_op.type == 'prelu':
            params = 1
        flops = 1
        for one_dim in in_data_shape[1:]:
            flops *= one_dim

    elif one_op.type == 'batch_norm':
        in_data_shape = block_vars[one_op.input("X")[0]].shape
        out_data_shape = block_vars[one_op.output("Y")[0]].shape
        _, c_in, h_out, w_out = in_data_shape
        # gamma, beta
        params = c_in * 2
        # compute mean and std
        flops = h_out * w_out * c_in * 2

    else:
        return None

    return in_data_shape, out_data_shape, params, flops


def _format_summary(collected_ops_list):
    '''
    Format summary report.
    Args:
        collected_ops_list: the collected operator with summary
    Returns:
        summary_table: summary report format
        total: sum param and flops
    '''
    summary_table = PrettyTable(
        ["No.", "TYPE", "INPUT", "OUTPUT", "PARAMs", "FLOPs"])
    summary_table.align = 'r'

    total = {}
    total_params = []
    total_flops = []
    for i, one_op in enumerate(collected_ops_list):
        # notice the order
        table_row = [
            i,
            one_op['type'],
            one_op['input_shape'],
            one_op['out_shape'],
            int(one_op['PARAMs']),
            int(one_op['FLOPs']),
        ]
        summary_table.add_row(table_row)
        total_params.append(int(one_op['PARAMs']))
        total_flops.append(int(one_op['FLOPs']))

    total['params'] = total_params
    total['flops'] = total_flops

    return summary_table, total


def _print_summary(summary_table, total):
    '''
    Print all the summary on terminal.
    Args:
        summary_table: summary report format
        total: sum param and flops
    '''
    parmas = total['params']
    flops = total['flops']
    print(summary_table)
    print('Total PARAMs: {}({:.4f}M)'.format(
        sum(parmas), sum(parmas) / (10 ** 6)))
    print('Total FLOPs: {}({:.2f}G)'.format(sum(flops), sum(flops) / 10 ** 9))
    print(
        "Notice: \n now supported ops include [Conv, DepthwiseConv, FC(mul), BatchNorm, Pool, Activation(sigmoid, tanh, relu, leaky_relu, prelu)]"
    )


def get_batch_dt_res(nmsed_out_v, data, contiguous_category_id_to_json_id, batch_size):
    dts_res = []
    lod = nmsed_out_v[0].lod()[0]
    nmsed_out_v = np.array(nmsed_out_v[0])
    real_batch_size = min(batch_size, len(data))
    assert (len(lod) == real_batch_size + 1), \
    "Error Lod Tensor offset dimension. Lod({}) vs. batch_size({})".format(len(lod), batch_size)
    k = 0
    for i in range(real_batch_size):
        dt_num_this_img = lod[i + 1] - lod[i]
        image_id = int(data[i][4][0])
        image_width = int(data[i][4][1])
        image_height = int(data[i][4][2])
        for j in range(dt_num_this_img):
            dt = nmsed_out_v[k]
            k = k + 1
            category_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            xmin = max(min(xmin, 1.0), 0.0) * image_width
            ymin = max(min(ymin, 1.0), 0.0) * image_height
            xmax = max(min(xmax, 1.0), 0.0) * image_width
            ymax = max(min(ymax, 1.0), 0.0) * image_height
            w = xmax - xmin
            h = ymax - ymin
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': image_id,
                'category_id': contiguous_category_id_to_json_id[category_id],
                'bbox': bbox,
                'score': score
            }
            dts_res.append(dt_res)
    return dts_res