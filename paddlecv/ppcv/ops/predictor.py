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
from paddle.inference import Config
from paddle.inference import create_predictor


class PaddlePredictor(object):
    def __init__(self,
                 param_path,
                 model_path,
                 config,
                 delete_pass=[],
                 name='model'):
        super().__init__()
        run_mode = config.get("run_mode", "paddle"),  # used trt or mkldnn
        shape_info_filename = os.path.join(
            config.get("output_dir", "output"),
            '{}_{}_dynamic_shape.txt'.format(name, run_mode))

        self.predictor, self.inference_config, self.input_names, self.input_tensors, self.output_tensors = self.create_paddle_predictor(
            param_path,
            model_path,
            batch_size=config['batch_size'],
            run_mode=run_mode,
            device=config.get("device", "CPU"),
            min_subgraph_size=config["min_subgraph_size"],
            shape_info_filename=shape_info_filename,
            trt_calib_mode=config["trt_calib_mode"],
            cpu_threads=config["cpu_threads"],
            trt_use_static=config["trt_use_static"],
            delete_pass=delete_pass)

    def create_paddle_predictor(self,
                                param_path,
                                model_path,
                                batch_size=1,
                                run_mode='paddle',
                                device='CPU',
                                min_subgraph_size=3,
                                shape_info_filename=None,
                                trt_calib_mode=False,
                                cpu_threads=6,
                                trt_use_static=False,
                                delete_pass=[]):
        if not os.path.exists(model_path) or not os.path.exists(param_path):
            raise ValueError(
                f"inference model: {model_path} or param: {param_path} does not exist, please check again..."
            )
        assert run_mode in [
            "paddle", "trt_fp32", "trt_fp16", "trt_int8", "mkldnn",
            "mkldnn_bf16"
        ], "The run_mode must be 'paddle', 'trt_fp32', 'trt_fp16', 'trt_int8', 'mkldnn', 'mkldnn_bf16', but received run_mode: {}".format(
            run_mode)
        config = Config(model_path, param_path)
        if device == 'GPU':
            config.enable_use_gpu(200, 0)
        else:
            config.disable_gpu()
            if 'mkldnn' in run_mode:
                try:
                    config.enable_mkldnn()
                    config.set_cpu_math_library_num_threads(cpu_threads)
                    if 'bf16' in run_mode:
                        config.enable_mkldnn_bfloat16()
                except Exception as e:
                    print(
                        "The current environment does not support `mkldnn`, so disable mkldnn."
                    )
                    pass

        precision_map = {
            'trt_int8': Config.Precision.Int8,
            'trt_fp32': Config.Precision.Float32,
            'trt_fp16': Config.Precision.Half
        }
        if run_mode in precision_map.keys():
            config.enable_tensorrt_engine(
                workspace_size=(1 << 25) * batch_size,
                max_batch_size=batch_size,
                min_subgraph_size=min_subgraph_size,
                precision_mode=precision_map[run_mode],
                use_static=trt_use_static,
                use_calib_mode=trt_calib_mode)

            if shape_info_filename is not None:
                if not os.path.exists(shape_info_filename):
                    config.collect_shape_range_info(shape_info_filename)
                    print(
                        f"collect dynamic shape info into : {shape_info_filename}"
                    )
                else:
                    print(
                        f"dynamic shape info file( {shape_info_filename} ) already exists, not need to generate again."
                    )
                config.enable_tuned_tensorrt_dynamic_shape(shape_info_filename,
                                                           True)

        # disable print log when predict
        config.disable_glog_info()
        for del_p in delete_pass:
            config.delete_pass(del_p)
        # enable shared memory
        config.enable_memory_optim()
        config.switch_ir_optim(True)
        # disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensors = []
        output_tensors = []
        for input_name in input_names:
            input_tensor = predictor.get_input_handle(input_name)
            input_tensors.append(input_tensor)
        output_names = predictor.get_output_names()
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
        return predictor, config, input_names, input_tensors, output_tensors

    def get_input_names(self):
        return self.input_names

    def run(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]

        for idx in range(len(x)):
            self.input_tensors[idx].copy_from_cpu(x[idx])
        self.predictor.run()
        result = []

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        for name in output_names:
            output = self.predictor.get_output_handle(name).copy_to_cpu()
            result.append(output)
        return result
