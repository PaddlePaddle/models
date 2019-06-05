import os
from paddle.fluid import layers
import paddle.fluid.core as core
from paddle.fluid.executor import Executor
from paddle.fluid.evaluator import Evaluator
from paddle.fluid.framework import Program, Parameter, default_main_program, default_startup_program, Variable, program_guard

def _clone_var_in_block_(block, var, name, dtype):
    assert isinstance(var, Variable)
    return block.create_var(
        name=name,
        shape=var.shape,
        dtype=dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True)

def load_vars(executor,
              dirname,
              main_program=None,
              vars=None,
              predicate=None,
              filename=None):
    
    load_dirname = os.path.normpath(dirname)
    if vars is None:
        if main_program is None:
            main_program = default_main_program()
        if not isinstance(main_program, Program):
            raise TypeError("program's type should be Program")

        load_vars(
            executor,
            dirname=load_dirname,
            main_program=main_program,
            vars=list(filter(predicate, main_program.list_vars())),
            filename=filename)
    else:
        load_prog = Program()
        load_block = load_prog.global_block()

        if main_program is None:
            main_program = default_main_program()
        if not isinstance(main_program, Program):
            raise TypeError("program should be as Program type or None")

        load_var_map = {}
        for each_var in vars:
            assert isinstance(each_var, Variable)
            if each_var.type == core.VarDesc.VarType.RAW:
                continue
            new_var = _clone_var_in_block_(load_block, each_var, each_var.name, each_var.dtype)
            if filename is None:
                if new_var.dtype == core.VarDesc.VarType.FP16:
                    new_var_master = _clone_var_in_block_(load_block,
                                                          each_var,
                                                          each_var.name+'.master',
                                                          core.VarDesc.VarType.FP32)
                    load_block.append_op(
                        type='load',
                        inputs={},
                        outputs={'Out': [new_var]},
                        attrs={
                            'file_path': os.path.join(load_dirname, new_var.name),
                            'load_as_fp16': True
                        })
                    load_block.append_op(
                        type='load',
                        inputs={},
                        outputs={'Out': [new_var_master]},
                        attrs={
                            'file_path': os.path.join(load_dirname, new_var.name),
                        })
                else:
                    load_block.append_op(
                        type='load',
                        inputs={},
                        outputs={'Out': [new_var]},
                        attrs={
                            'file_path': os.path.join(load_dirname, new_var.name)
                        })
            else:
                load_var_map[new_var.name] = new_var

        if filename is not None:
            load_var_list = []
            for name in sorted(load_var_map.keys()):
                load_var_list.append(load_var_map[name])

            load_block.append_op(
                type='load_combine',
                inputs={},
                outputs={"Out": load_var_list},
                attrs={'file_path': os.path.join(load_dirname, filename)})
        executor.run(load_prog)
