import paddle.fluid.core as core
import numpy as np
import math
import os

U8_MAX = 255
S8_MAX = 127

def dot(program, output_name="model.dot"):
    dot_graph = ""
    dot_nodes = []
    dot_edges = []
    dot_graph += "digraph pm {\n"
    for block in program.blocks:
        ops = list(block.ops)
        block_id = block.idx
        for op in ops:
            op_type = op.type
            op_name = op_type + "_" + op.output_arg_names[0].replace(".", "_")
            for name in op.input_arg_names:
                name = name.replace(".", "_")
                dot_edge = name + " -> " + op_name
                if dot_edge not in dot_edges:
                    dot_edges.append(dot_edge)
                dot_node = name + " [shape=oval, style=filled, fillcolor=yellow]"
                if dot_node not in dot_nodes:
                    dot_nodes.append(dot_node)

            for name in op.output_arg_names:
                name = name.replace(".", "_")
                dot_edge = op_name + " -> " + name
                if dot_edge not in dot_edges:
                    dot_edges.append(dot_edge)
            if op_type in ["conv2d", "pool2d", "quantize", "dequantize"]:
                dot_node = op_name + " [shape=box, style=filled, color=greenyellow]"
            else:
                dot_node = op_name + " [shape=box, style=filled, fillcolor=red]"

            if dot_node not in dot_nodes:
                dot_nodes.append(dot_node)

    for dot_edge in dot_edges:
        dot_graph += dot_edge + "\n"
    for dot_node in dot_nodes:
        dot_graph += dot_node + "\n"
    dot_graph += "}"

    with open(output_name, 'w') as f:
        f.write(dot_graph)

def get_quantize_dequantize_combination(program):
    conv_op_index = [index for index, value in enumerate(program.global_block().ops) if value.type == 'conv2d']

    if len(conv_op_index) < 2:
        return []
    res = []
    for index, value in enumerate(conv_op_index):
        if index == 0: 
            res.append(conv_op_index[1] - 1) # first quantize op
        elif index == len(conv_op_index) - 1:
            res.append(conv_op_index[-1])
        else:
            if value + 1 < conv_op_index[index + 1]:
                for i in range(value + 1, conv_op_index[index + 1]):
                    cur_op_type = program.current_block().ops[i].type
                    if cur_op_type in ["conv2d"] and i < conv_op_index[index + 1] :
                        continue
                    else:
                        res.append(i - 1)
                        break
                for i in range(conv_op_index[index + 1] - 1, res[-1] - 1, -1):
                    cur_op_type = program.current_block().ops[i].type
                    if cur_op_type not in ["conv2d"]:
                        res.append(i)
                        break
                    else:
                        continue
            else:
                continue
    return res

def update_program_for_saving_var(program, save_prog,  name, value, data_shape, dst, data_type="float32"):
    tmp_var = program.current_block().create_var(
        name=name,
        dtype=data_type,
        persistable=True,
    )

    program.current_block().append_op(
        type='assign_value',
        outputs={'Out': [tmp_var]},
        attrs={
            'dtype':core.VarDesc.VarType.FP32,
            'shape': data_shape,
            'fp32_values': value
        }
    )

    program.current_block().append_op(
        type = 'save',
        inputs={'X': '{}'.format(name)},
        outputs={},
        attrs={"file_path": "{}/{}".format(dst, name)}
    )

    save_prog.current_block().create_var(
        name=name,
        dtype=data_type,
        persistable=True,
    )

def check_op_ancestor(program, var_name):
    search_end_index = 0
    input_index_name = {}
    output_index_name = {}
    ops_type = []
    found_var = False
    first_conv_op_index = -1
    first_conv_op_flag = False
    for index, op in enumerate(program.current_block().ops):
        input_var_list = []
        output_var_list = []
        ops_type.append(op.type)
        if op.type == 'conv2d' and not first_conv_op_flag:
            first_conv_op_index = index
            first_conv_op_flag = True
        op_input_name = op.input_names[0]
        input_var_list.append(op.input(op_input_name)[0])
        input_index_name[index] = input_var_list
        
        for name in op.output_names:
            output_var_list.append(op.output(name)[0])

            if op.output(name)[0] == var_name:
                search_end_index = index
                found_var = True
                break
        output_index_name[index] = output_var_list

        if found_var: break
    # analysis
    while search_end_index >= 0:
        input_name = input_index_name[search_end_index][0]
        found_ancestor = False
        for i in output_index_name.keys():
            if input_name in output_index_name[i]:
                search_end_index = i
                found_ancestor = True
                break

        if not found_ancestor: # Dangling var
            return S8_MAX
        elif search_end_index == first_conv_op_index: # first conv
            if program.current_block().ops[i].has_attr('fuse_relu') and program.current_block().ops[i].attr('fuse_relu'):
                return U8_MAX
            else:
                return S8_MAX
        elif ops_type[search_end_index] != 'conv2d':
            continue
        else:
            if program.current_block().ops[i].has_attr('fuse_relu') and program.current_block().ops[i].attr('fuse_relu'):
                return U8_MAX
            else:
                return S8_MAX
    return S8_MAX


def update_program_attr(program, attr="is_test", value=True):
    special_op_type_list = [
                    "pool2d", "sigmoid", "logsigmoid", "softshrink", "exp",
                    "brelu", "pow", "leaky_relu", "stanh", "relu", "tanh",
                    "tanh_shrink", "sqrt", "abs", "ceil", "elu", "floor", "cos",
                    "sin", "round", "reciprocal", "hard_shrink", "hard_sigmoid",
                    "relu6", "soft_relu", "swish", "thresholded_relu", "log",
                    "square", "softplus", "softsign"]
    for i in program.current_block().ops:
        if i.has_attr(attr) or i.type in special_op_type_list:
            i._set_attr(attr, value)
    program = program.clone()

def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def expand_quantized_bins(quantized_bins, reference_bins):
    expanded_quantized_bins = [0]*len(reference_bins)
    num_merged_bins = len(reference_bins)/len(quantized_bins)
    j_start = 0
    j_end = num_merged_bins
    for idx in xrange(len(quantized_bins)):
        zero_count = reference_bins[j_start:j_end].count(0)
        num_merged_bins = j_end-j_start
        if zero_count == num_merged_bins:
            avg_bin_ele = 0
        else:
            avg_bin_ele = quantized_bins[idx]/(num_merged_bins - zero_count + 0.0)
        for idx1 in xrange(j_start, j_end):
            expanded_quantized_bins[idx1] = (0 if reference_bins[idx1] == 0 else avg_bin_ele)
        j_start += num_merged_bins
        j_end += num_merged_bins
        if (idx+1) == len(quantized_bins) - 1:
            j_end = len(reference_bins)
    return expanded_quantized_bins

def safe_entropy(reference_distr_P, P_sum, candidate_distr_Q, Q_sum):
    assert len(reference_distr_P) == len(candidate_distr_Q)
    tmp_sum1 = 0
    tmp_sum2 = 0
    for idx in range(len(reference_distr_P)):
        p_idx = reference_distr_P[idx]
        q_idx = candidate_distr_Q[idx]
        if p_idx == 0:
            tmp_sum1 += 0
            tmp_sum2 += 0
        else:
            if q_idx == 0:
	        print "Fatal error!, idx = " + str(idx) + " qindex = 0! p_idx = " + str(p_idx)
            tmp_sum1 += p_idx * (math.log(Q_sum*p_idx))
            tmp_sum2 += p_idx * (math.log(P_sum*q_idx))
    return (tmp_sum1 - tmp_sum2)/P_sum

def get_optimal_scaling_factor(activation_blob, num_quantized_bins = 255):
    max_val = np.max(activation_blob)
    min_val = np.min(activation_blob)
    # print min_val, max_val
    if min_val >= 0:
        hist, hist_edeges = np.histogram(activation_blob, bins=2048, range=(min_val, max_val))
        ending_iter = 2047
        starting_iter = int(ending_iter * 0.7)
    else:
        th = max(abs(max_val), abs(min_val))
        hist, hist_edeges = np.histogram(activation_blob, bins=2048, range=(-th, th))
        starting_iter = 0
        ending_iter = 2047
        if abs(max_val) > abs(min_val):
            while starting_iter < ending_iter:
                if hist[starting_iter] == 0:
                    starting_iter += 1
                    continue
                else:
                    break
            starting_iter += int((ending_iter - starting_iter)*0.6)
        else:
            while ending_iter > 0:
                if hist[ending_iter] == 0:
                    ending_iter -= 1
                    continue
                else:
                    break
            starting_iter = int(0.6 * ending_iter)
    bin_width = hist_edeges[1]-hist_edeges[0]
    P_sum = len(activation_blob)
    min_kl_divergence = 0
    min_kl_index = 0
    kl_inited = False
    for i in range(starting_iter, ending_iter+1):
        reference_distr_P = hist[0:i].tolist()
        outliers_count = sum(hist[i:2048])
        if reference_distr_P[i-1] == 0:
            continue
        reference_distr_P[i-1] += outliers_count
        reference_distr_bins = reference_distr_P[:]
        candidate_distr_Q = hist[0:i].tolist()
        num_merged_bins = i/num_quantized_bins
        candidate_distr_Q_quantized = [0]*num_quantized_bins
        j_start = 0
        j_end = num_merged_bins
        for idx in xrange(num_quantized_bins):
	    candidate_distr_Q_quantized[idx] = sum(candidate_distr_Q[j_start:j_end])
            j_start += num_merged_bins
            j_end += num_merged_bins
            if (idx+1) == num_quantized_bins - 1:
	        j_end = i
	candidate_distr_Q = expand_quantized_bins(candidate_distr_Q_quantized, reference_distr_bins)
        Q_sum = sum(candidate_distr_Q)
        kl_divergence = safe_entropy(reference_distr_P, P_sum, candidate_distr_Q, Q_sum)
        if not kl_inited:
            min_kl_divergence = kl_divergence
            min_kl_index = i
            kl_inited = True
        elif kl_divergence < min_kl_divergence:
	    min_kl_divergence = kl_divergence
            min_kl_index = i
	else:
	    pass
    if min_kl_index == 0:
        while starting_iter > 0:
            if hist[starting_iter] == 0:
                starting_iter -= 1
                continue
            else:
                break
        min_kl_index = starting_iter
    return (min_kl_index+0.5)*bin_width

def analysis_program_var(program):
    conv_op_index = [index for index, value in enumerate(program.global_block().ops) if value.type == 'conv2d']
    weights_var_name = []
    conv_input_var_name = []
    conv_output_var_name = []
    quantization_var_name = []
    
    for i in conv_op_index[1:]:
        weights_var_name.append(program.current_block().ops[i].input('Filter')[0])
        conv_input_var_name.append(program.current_block().ops[i].input('Input')[0])
        conv_output_var_name.append(program.current_block().ops[i].output('Output')[0])
    
    quantize_dequantize_index = get_quantize_dequantize_combination(program)
    
    for i in range(0, len(quantize_dequantize_index), 2):
        output_index = quantize_dequantize_index[0] - 1
        input_index = quantize_dequantize_index[1] 
        for j in range(quantize_dequantize_index[1] + 1, len(program.current_block().ops) - 1):
            if program.current_block().ops[j].type in ("pool2d") and program.current_block().ops[j + 1].type not in ("pool2d"):
                input_index = j
                break
            else:
                continue
        output_var_name = program.current_block().ops[output_index].output_names[0]
        output_var = program.current_block().ops[output_index].output(output_var_name)[0]
        input_var_name = program.current_block().ops[input_index].output_names[0]
        input_var = program.current_block().ops[input_index].output(input_var_name)[0]
        quantization_var_name.append(output_var)
        quantization_var_name.append(input_var)
    
    return conv_input_var_name, conv_output_var_name, weights_var_name, quantization_var_name

def check_op_type_by_input(program, var_name, start_index=0):
    op_type_list = []
    for op in program.current_block().ops[start_index: ]:
        for input_key in op.input_names:
            if op.input(input_key) and op.input(input_key)[0] == var_name:
                op_type_list.append(op.type)
    for i in op_type_list:
        if not i in ['conv2d', 'pool2d']:
            return False
    return True
