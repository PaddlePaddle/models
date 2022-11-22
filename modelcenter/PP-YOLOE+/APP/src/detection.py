import cv2
import os
import numpy as np
import yaml
from paddle.inference import Config, create_predictor, PrecisionType
from PIL import Image

from .download import get_model_path
from .preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, decode_image
from .visualize import draw_det

class Detector(object):
    def __init__(self, model_name):
        parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
        yml_file = os.path.join(parent_path, 'configs/{}.yml'.format(model_name))
        with open(yml_file, 'r') as f:
            yml_conf = yaml.safe_load(f)
        
        infer_model = get_model_path(yml_conf['model_path'])
        infer_params = get_model_path(yml_conf['param_path'])
        config = Config(infer_model, infer_params)
        device = yml_conf.get('device', 'CPU')
        run_mode = yml_conf.get('mode', 'paddle')
        cpu_threads = yml_conf.get('cpu_threads', 1)
        if device == 'CPU':
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(cpu_threads)
        elif device == 'GPU':
            # initial GPU memory(M), device ID
            config.enable_use_gpu(200, 0)
            # optimize graph and fuse op
            config.switch_ir_optim(True)

        precision_map = {
            'trt_int8': Config.Precision.Int8,
            'trt_fp32': Config.Precision.Float32,
            'trt_fp16': Config.Precision.Half
        }


        if run_mode in precision_map.keys():
            config.enable_tensorrt_engine(
                workspace_size=(1 << 25) * batch_size,
                max_batch_size=batch_size,
                min_subgraph_size=yml_conf['min_subgraph_size'],
                precision_mode=precision_map[run_mode],
                use_static=True,
                use_calib_mode=False)

            if yml_conf['use_dynamic_shape']:
                min_input_shape = {
                    'image': [batch_size, 3, 640, 640],
                    'scale_factor': [batch_size, 2]
                }
                max_input_shape = {
                    'image': [batch_size, 3, 1280, 1280],
                    'scale_factor': [batch_size, 2]
                }
                opt_input_shape = {
                    'image': [batch_size, 3, 1024, 1024],
                    'scale_factor': [batch_size, 2]
                }
                config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                                opt_input_shape)
        
        # disable print log when predict
        config.disable_glog_info()
        # enable shared memory
        config.enable_memory_optim()
        # disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
        self.predictor = create_predictor(config)
        self.yml_conf = yml_conf
        self.preprocess_ops = self.create_preprocess_ops(yml_conf)
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        self.draw_threshold = yml_conf.get('draw_threshold', 0.5)
        self.class_names = yml_conf['label_list']

    
    def create_preprocess_ops(self, yml_conf):
        preprocess_ops = []
        for op_info in yml_conf['Preprocess']:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops
    
    def create_inputs(self, image_files):
        inputs = dict()
        im_list, im_info_list = [], []
        for im_path in image_files:
            im, im_info = preprocess(im_path, self.preprocess_ops)
            im_list.append(im)
            im_info_list.append(im_info)

        inputs['im_shape'] = np.stack([e['im_shape'] for e in im_info_list], axis=0).astype('float32')
        inputs['scale_factor'] = np.stack([e['scale_factor'] for e in im_info_list], axis=0).astype('float32')
        inputs['image'] = np.stack(im_list, axis=0).astype('float32')
        return inputs
    
    def __call__(self, image_file):
        inputs = self.create_inputs([image_file])
        for name in self.input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])
        
        self.predictor.run()
        boxes_tensor = self.predictor.get_output_handle(self.output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        boxes_num = self.predictor.get_output_handle(self.output_names[1])
        np_boxes_num = boxes_num.copy_to_cpu()
        if np_boxes_num.sum() <= 0:
            np_boxes = np.zeros([0, 6])
        
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        elif isinstance(image_file, np.ndarray):
            image = image_file
        expect_boxes = (np_boxes[:, 1] > self.draw_threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        image = draw_det(image, np_boxes, self.class_names)
        return image, {'bboxes': np_boxes.tolist()}