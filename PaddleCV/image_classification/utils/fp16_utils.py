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

def create_master_params_grads(params_grads, main_prog, startup_prog, scale_loss, reduce_master_grad=True):
    master_params_grads = []      # master p, g on local device
    params_grads_to_apply = []    # master p, g after allreduced, if reduce_master_grad is enabled
    tmp_role = main_prog._current_role
    OpRole = fluid.core.op_proto_and_checker_maker.OpRole
    main_prog._current_role = OpRole.Backward
    for p, g in params_grads:
        # create master parameters
        master_param = copy_to_master_param(p, main_prog.global_block())
        startup_master_param = startup_prog.global_block()._clone_variable(master_param)
        startup_p = startup_prog.global_block().var(p.name)
        cast_fp16_to_fp32(startup_p, startup_master_param, startup_prog)
        # cast fp16 gradients to fp32 before apply gradients
        if g.name.startswith("batch_norm"):
            if scale_loss > 1:
                scaled_g = g / float(scale_loss)
            else:
                scaled_g = g
            master_params_grads.append([p, scaled_g])
            continue

        master_grad = fluid.layers.cast(g, "float32")
        if scale_loss > 1:
            master_grad = master_grad / float(scale_loss)
        master_params_grads.append([p, master_grad])
        if reduce_master_grad:
            reduced_master_grad = fluid.layers.collective._allreduce(master_grad)
        else:
            reduced_master_grad = master_grad
        params_grads_to_apply.append([master_param, reduced_master_grad])
    
    # update program op role var acording to master grads before allreduce.
    _update_role_var_grad(main_prog, master_params_grads)
    main_prog._current_role = tmp_role
    return params_grads_to_apply

def master_param_to_train_param(master_params_grads, params_grads, main_prog):
    for idx, m_p_g in enumerate(master_params_grads):
        with main_prog._optimized_guard([m_p_g[0], m_p_g[1]]):
            train_p_name = m_p_g[0].name.replace(".master", "")
            if train_p_name.startswith("batch_norm"):
                continue
            train_p = None
            # find fp16 param in original params_grads list
            for p, g in params_grads:
                if p.name == train_p_name:
                    train_p = p
            if not train_p:
                print("can not find train param for: ", m_p_g[0].name)
                continue
            cast_fp32_to_fp16(m_p_g[0], train_p, main_prog)
