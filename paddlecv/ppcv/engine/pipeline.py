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
import sys
import numpy as np
import math
import glob
import paddle
import cv2
from collections import defaultdict
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from ppcv.core.framework import Executor
from ppcv.utils.logger import setup_logger
from ppcv.core.config import ConfigParser

logger = setup_logger('pipeline')

__all__ = ['Pipeline']


class Pipeline(object):
    def __init__(self, cfg):
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()
        self.exe = Executor(self.model_cfg, self.env_cfg)
        self.output_dir = self.env_cfg.get('output_dir', 'output')

    def _parse_input(self, input):
        if isinstance(input, np.ndarray):
            return [input], 'data'
        if isinstance(input, Sequence) and isinstance(input[0], np.ndarray):
            return input, 'data'
        im_exts = ['jpg', 'jpeg', 'png', 'bmp']
        im_exts += [ext.upper() for ext in im_exts]
        video_exts = ['mp4', 'avi', 'wmv', 'mov', 'mpg', 'mpeg', 'flv']
        video_exts += [ext.upper() for ext in video_exts]

        if isinstance(input, (list, tuple)) and isinstance(input[0], str):
            input_type = "image"
            images = [
                image for image in input
                if any([image.endswith(ext) for ext in im_exts])
            ]
            return images, input_type

        if os.path.isdir(input):
            input_type = "image"
            logger.info(
                'Input path is directory, search the images automatically')
            images = set()
            infer_dir = os.path.abspath(input)
            for ext in im_exts:
                images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
            images = list(images)
            return images, input_type

        logger.info('Input path is {}'.format(input))
        input_ext = os.path.splitext(input)[-1][1:]
        if input_ext in im_exts:
            input_type = "image"
            return [input], input_type

        if input_ext in video_exts:
            input_type = "video"
            return input, input_type

        raise ValueError("Unsupported input format: {}".fomat(input_ext))
        return

    def run(self, input):
        input, input_type = self._parse_input(input)
        if input_type == "image" or input_type == 'data':
            results = self.predict_images(input)
        elif input_type == "video":
            results = self.predict_video(input)
        else:
            raise ValueError("Unexpected input type: {}".format(input_type))
        return results

    def decode_image(self, input):
        if isinstance(input, str):
            with open(input, 'rb') as f:
                im_read = f.read()
            data = np.frombuffer(im_read, dtype='uint8')
            im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = input
        return im

    def predict_images(self, input):
        batch_input = [{
            'input.image': self.decode_image(f),
            'input.fn': 'tmp.jpg' if isinstance(f, np.ndarray) else f
        } for f in input]
        results = self.exe.run(batch_input)
        return results

    def predict_video(self, input):
        capture = cv2.VideoCapture(input)
        file_name = input.split('/')[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("video fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, file_name)
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        frame_id = 0

        results = None
        while (1):
            if frame_id % 10 == 0:
                logger.info('frame id: {}'.format(frame_id))
            ret, frame = capture.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_input = [{'input.image': frame_rgb, 'input.fn': input}]
            results = self.exe.run(frame_input, frame_id)
            writer.write(results[0]['output'])
            frame_id += 1
        writer.release()
        logger.info('save result to {}'.format(out_path))
        return results
