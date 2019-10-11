from __future__ import print_function
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


def cast_fp16_to_fp32(i, o, prog):
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={
            "in_dtype": fluid.core.VarDesc.VarType.FP16,
            "out_dtype": fluid.core.VarDesc.VarType.FP32
        }
    )


def cast_fp32_to_fp16(i, o, prog):
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={
            "in_dtype": fluid.core.VarDesc.VarType.FP32,
            "out_dtype": fluid.core.VarDesc.VarType.FP16
        }
    )


def copy_to_master_param(p, block):
    v = block.vars.get(p.name, None)
    if v is None:
        raise ValueError("no param name %s found!" % p.name)
    new_p = fluid.framework.Parameter(
        block=block,
        shape=v.shape,
        dtype=fluid.core.VarDesc.VarType.FP32,
        type=v.type,
        lod_level=v.lod_level,
        stop_gradient=p.stop_gradient,
        trainable=p.trainable,
        optimize_attr=p.optimize_attr,
        regularizer=p.regularizer,
        gradient_clip_attr=p.gradient_clip_attr,
        error_clip=p.error_clip,
        name=v.name + ".master")
    return new_p


def _update_role_var_grad(prog, params_grads):
    BACKWARD = core.op_proto_and_checker_maker.OpRole.Backward
    gradname_to_paramname = dict()
    for p, g in params_grads:
        gradname_to_paramname[g.name] = p.name
    for op in prog.global_block().ops:
        role = op.attr("op_role")
        if role & int(BACKWARD) and op.has_attr("op_role_var"):
            # have backward bits then remove all op_role_var
            op.desc.remove_attr("op_role_var")
    for op in prog.global_block().ops:
        if op.type == "allreduce":
            allreduce_role_var = []
            for input_varname in op.input_arg_names:
                if input_varname in gradname_to_paramname:
                    allreduce_role_var.append(gradname_to_paramname[input_varname])
                    allreduce_role_var.append(input_varname)
            print("updating role var: ", allreduce_role_var)
            op._set_attr("op_role_var", allreduce_role_var)


#def create_master_params_grads(params_grads, main_prog, startup_prog, scale_loss, reduce_master_grad=True):
def create_master_params_grads(params_grads, main_prog, startup_prog, scale_loss, reduce_master_grad=False):
    master_params_grads = []      # master p, g on local device
    #params_grads_to_apply = []    # master p, g after allreduced, if reduce_master_grad is enabled
    #tmp_role = main_prog._current_role
    #OpRole = fluid.core.op_proto_and_checker_maker.OpRole
    #main_prog._current_role = OpRole.Backward
    with main_prog._backward_role_guard():
        for p, g in params_grads:
            # create master parameters
            master_param = copy_to_master_param(p, main_prog.global_block())
            startup_master_param = startup_prog.global_block()._clone_variable(master_param)
            startup_p = startup_prog.global_block().var(p.name)
            # fp16 -> fp32
            cast_fp16_to_fp32(startup_p, startup_master_param, startup_prog)
            # cast fp16 gradients to fp32 before apply gradients
            if g.name.find("batch_norm") > -1:
                scaled_g = g / scale_loss
                master_params_grads.append([p, scaled_g])
                continue

            master_grad = fluid.layers.cast(g, "float32")
            master_grad = master_grad / scale_loss
            master_params_grads.append([master_param, master_grad])

    return master_params_grads


def master_param_to_train_param(master_params_grads, params_grads, main_prog):
    for idx, m_p_g in enumerate(master_params_grads):
        train_p, _ = params_grads[idx]
        if train_p.name.find('batch_norm') > -1:
            continue
        with main_prog._optimized_guard([m_p_g[0], m_p_g[1]]):
            # fp32 -> fp16
            cast_fp32_to_fp16(m_p_g[0], train_p, main_prog)
