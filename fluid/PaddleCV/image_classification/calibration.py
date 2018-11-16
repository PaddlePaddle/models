from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import models
import reader
import argparse
import functools
from utility import add_arguments, print_arguments
import paddle.fluid.core as core
import os
sys.path.append('..')
import int8.utility as ut
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  32,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('out',              str,  "calibration_out",   "Output INT8 model")
add_arg('with_mem_opt',     bool, True,                "Whether to use memory optimization or not.")
add_arg('use_train_data',   bool, False,               "Whether to use train data for sampling or not.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('model',            str, "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('iterations',       int, 1, "Sampling iteration")
add_arg('algo',             str, 'direct', "calibration algo")
add_arg('debug',            bool, False, "print program and save the dot file")

# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def eval(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    int8_model = os.path.join(os.getcwd(), args.out)
    print ("Start calibration for {}...".format(model_name))
   
    tmp_scale_folder = ".tmp_scale"
    
    if os.path.exists(int8_model): # Not really need to execute below operations
        os.system("rm -rf " + int8_model)
        os.system("mkdir " + int8_model)
    
    if not os.path.exists(tmp_scale_folder):
        os.system("mkdir {}".format(tmp_scale_folder))

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = models.__dict__[model_name]()
    
    if model_name is "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
        out = model.net(input=image, class_dim=class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)

        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    test_program = fluid.default_main_program().clone(for_test=True)
    
    if with_memory_optimization:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    t = fluid.transpiler.InferenceTranspiler()
    t.transpile(test_program, fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace())

    conv_op_index = [index for index, value in enumerate(test_program.global_block().ops) if value.type == 'conv2d']

    conv_input_var_name, conv_output_var_name, weights_var_name, quant_dequant_var_name = ut.analysis_program_var(test_program)

    persitable_vars = []
    for i in test_program.list_vars():
        if not i.persistable and i.name in weights_var_name or i.name in conv_input_var_name or i.name in conv_output_var_name or i.name in quant_dequant_var_name:
            i.persistable= True
            persitable_vars.append(i.name)
    int8_prog = test_program.clone()          

    sampling_reader = paddle.batch(reader.train() if args.use_train_data else reader.val(), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    var_max_value_map = {}
    var_max_range = {}
    u8_output_var = []
    s8_output_var = []
    # Step 0 Samling the data
    for batch_id, data in enumerate(sampling_reader()):
        _, _, _ = exe.run(test_program,
                                   fetch_list=fetch_list,
                                   feed=feeder.feed(data))

        for i in test_program.list_vars():
            if i.name not in weights_var_name and i.name not in conv_input_var_name \
                and i.name not in conv_output_var_name and i.name not in quant_dequant_var_name:
                continue

            np_data = np.array(fluid.global_scope().find_var(i.name).get_tensor())

            if i.name in weights_var_name:
                max_value = []
                for j in range(np_data.shape[0]):
                    if not ut.is_close(float(np.max(np.abs(np_data[j]))), 0.0):
                       max_value.append(ut.S8_MAX/float(np.max(np.abs(np_data[j]))))
                    else:
                       max_value.append(0.0)
                max_range = ut.S8_MAX
            else:
                if i.name in conv_output_var_name:
                    cur_op = test_program.current_block().ops[conv_output_var_name.index(i.name) + 2]
                    if cur_op.has_attr('fuse_relu') and cur_op.attr('fuse_relu'):
                        max_range = ut.U8_MAX
                        u8_output_var.append(i.name)
                    else:
                        max_range = ut.S8_MAX
                        s8_output_var.append(i.name)
                else:
                    max_range = ut.check_op_ancestor(test_program, i.name)

                max_value = [np.abs(np_data)]
            
            var_max_range[i.name] = max_range

            if i.name not in var_max_value_map:
                var_max_value_map[i.name] = []
            
            var_max_value_map[i.name].append(max_value)
        
        if batch_id != args.iterations -1:
            continue
        
        break

    infer_prog = test_program.clone()

    func = ut.get_optimal_scaling_factor if args.algo == 'KL' else np.max
    for i in conv_input_var_name:
        ut.update_program_for_saving_var(infer_prog, int8_prog, i+"_scale.input.test",
         [var_max_range[i]/func(var_max_value_map[i])], np.array(np.max(var_max_value_map[i])).shape, tmp_scale_folder)
    
    for i in conv_output_var_name:
        ut.update_program_for_saving_var(infer_prog, int8_prog, i+"_scale.output.test", 
        [var_max_range[i]/func(var_max_value_map[i])], np.array(np.max(var_max_value_map[i])).shape, tmp_scale_folder)
    
    for i in weights_var_name:
        ut.update_program_for_saving_var(infer_prog, int8_prog, i+"_scale.weights.test",
         var_max_value_map[i][0], np.array((var_max_value_map[i][0])).shape, tmp_scale_folder)

    for i in quant_dequant_var_name:
        ut.update_program_for_saving_var(infer_prog, int8_prog, i+"_scale.output.test", 
        [var_max_range[i]/func(var_max_value_map[i])], np.array(np.max(var_max_value_map[i])).shape, tmp_scale_folder)

    feeded_var_names = []
    # Step 1 Save the scale variables
    for batch_id, data in enumerate(sampling_reader()):
        _, _, _ = exe.run(infer_prog,
                                   fetch_list=fetch_list,
                                   feed=feeder.feed(data))
        feeded_var_names = feeder.feed(data).keys()
        break

    for index, value in enumerate(conv_op_index[1:]):
        int8_prog.current_block().ops[value].desc.set_input("Scale_in", ["{}_scale.input.test".format(conv_input_var_name[index])])
        int8_prog.current_block().ops[value].desc.set_input("Scale_out", ["{}_scale.output.test".format(conv_output_var_name[index])])
        int8_prog.current_block().ops[value].desc.set_input("Scale_weights", ["{}_scale.weights.test".format(weights_var_name[index])])
        if int8_prog.current_block().ops[value].desc.input("ResidualData"):
            name = int8_prog.current_block().ops[value].desc.input("ResidualData")[0]
            int8_prog.current_block().ops[value].desc.set_input("Scale_in_eltwise", ["{}_scale.output.test".format(name)])
    
    # Step 2 Update the program with the quantization/dequantization op insertion. 

    insert_op_index = ut.get_quantize_dequantize_combination(int8_prog)

    inserted_op_length = 0
    has_other_int8_layer = False

    for i in range(0, len(insert_op_index), 2):
        quantize_tmp = int8_prog.current_block().create_var(
            name="quantize_{}_tmp".format(i),
            dtype=core.VarDesc.VarType.UINT8,
        )

        original_out_name = int8_prog.current_block().ops[insert_op_index[i] + inserted_op_length].output_names[0]

        original_out = int8_prog.current_block().ops[insert_op_index[i] + inserted_op_length].output(original_out_name)[0]

        op = int8_prog.current_block()._insert_op(
            index=insert_op_index[i] + inserted_op_length + 1,
            type="quantize",
            inputs={"Input": original_out,
                    "Scale": "{}_scale.input.test".format(conv_input_var_name[i])},
            outputs={"Output": quantize_tmp},
        )
        op._set_attr("data_format", "MKLDNNLAYOUT")
        op._set_attr("use_mkldnn", 1)
        inserted_op_length += 1

        for op in int8_prog.current_block().ops[insert_op_index[i] + inserted_op_length + 1:]:
            for j in op.input_names:
                if len(op.input(j)) and op.input(j)[0] == original_out and op.type in ["pool2d", "conv2d"]:
                    op.desc.set_input(j, ["{}".format(quantize_tmp.name)])

        start_index = insert_op_index[i + 1]  + inserted_op_length 
        end_index = len(int8_prog.current_block().ops) if i + 1 == len(insert_op_index) - 1 else insert_op_index[i + 1 + 1]
        while start_index < end_index:
            if int8_prog.current_block().ops[start_index + 1].type in ["pool2d"]:
                prev_index = start_index
                prev_output_key_name = int8_prog.current_block().ops[prev_index].output_names[0]
                prev_output_var_name = int8_prog.current_block().ops[prev_index].output(prev_output_key_name)[0]
                if not ut.check_op_type_by_input(int8_prog, prev_output_var_name, prev_index):
                    break
                start_index += 1
                has_other_int8_layer = True
                continue
            else:
                break

        dequantize_tmp_var = int8_prog.current_block().create_var(
            name="dequantize_{}_tmp".format(i + 1),
            dtype="float32",
            )

        original_out_name = int8_prog.current_block().ops[start_index].output_names[0]

        original_out = int8_prog.current_block().ops[start_index].output(original_out_name)[0]
        start_index += 1
        op = int8_prog.current_block()._insert_op(
            index= start_index,
            type= "dequantize",
            inputs={"Input": original_out,
                    "Scale": "{}_scale.output.test".format(original_out)},
            outputs={"Output": dequantize_tmp_var},
        )

        for index in range(start_index + 1, len(int8_prog.current_block().ops)):
            for j in int8_prog.current_block().ops[index].input_names:
                if len(int8_prog.current_block().ops[index].input(j)) and int8_prog.current_block().ops[index].input(j)[0] == original_out:
                    int8_prog.current_block().ops[index].desc.set_input(j, ["{}".format(dequantize_tmp_var.name)])
        
        inserted_op_length += 1

        op._set_attr("data_format", "MKLDNNLAYOUT")
        op._set_attr("use_mkldnn", 1)
        has_other_int8_layer = False  

    # Step 3 Save the new model 

    for i in int8_prog.list_vars():
        if i.name in persitable_vars:
            i.persitable = False
            os.system("rm -rf {}/{}".format(pretrained_model, i.name))

    for i in u8_output_var:
        int8_prog.current_block().var(i).desc.set_dtype(core.VarDesc.VarType.UINT8)
    
    for i in s8_output_var:
        int8_prog.current_block().var(i).desc.set_dtype(core.VarDesc.VarType.INT8)
    
    ut.update_program_attr(int8_prog)

    fluid.io.save_inference_model(int8_model, feeded_var_names, [int8_prog.current_block().var(i) for i in fetch_list], exe, int8_prog)
    
    if args.debug:
        ut.dot(int8_prog)
        print int8_prog

    print ("Calibration is done and the corresponding files were generated at {}".format(os.path.abspath(args.out)))

def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()
