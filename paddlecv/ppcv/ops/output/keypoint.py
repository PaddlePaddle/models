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
import glob
import paddle
import cv2
import json
from collections import defaultdict
from .base import OutputBaseOp
from ppcv.utils.logger import setup_logger
from ppcv.core.workspace import register

logger = setup_logger('KptOutput')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def draw_kpt(image, keypoints, visual_thresh=0.6, ids=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        plt.switch_backend('agg')
    except Exception as e:
        print('Matplotlib not found, please install matplotlib.'
              'for example: `pip install matplotlib`.')
        raise e
    image = image[:, :, ::-1]
    skeletons = np.array(keypoints)[0]
    kpt_nums = 17
    if len(skeletons) > 0:
        kpt_nums = skeletons.shape[1]
    if kpt_nums == 17:  #plot coco keypoint
        EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
                 (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14),
                 (13, 15), (14, 16), (11, 12)]
    else:  #plot mpii keypoint
        EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (3, 6), (6, 7),
                 (7, 8), (8, 9), (10, 11), (11, 12), (13, 14), (14, 15),
                 (8, 12), (8, 13)]
    NUM_EDGES = len(EDGES)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    plt.figure()

    color_set = None

    canvas = image.copy()
    for i in range(kpt_nums):
        for j in range(len(skeletons)):
            if skeletons[j][i, 2] < visual_thresh:
                continue
            if ids is None:
                color = colors[i] if color_set is None else colors[color_set[j]
                                                                   %
                                                                   len(colors)]
            else:
                color = get_color(ids[j])

            cv2.circle(
                canvas,
                tuple(skeletons[j][i, 0:2].astype('int32')),
                2,
                color,
                thickness=-1)

    to_plot = cv2.addWeighted(image, 0.3, canvas, 0.7, 0)
    fig = matplotlib.pyplot.gcf()

    stickwidth = 2

    for i in range(NUM_EDGES):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            if skeletons[j][edge[0], 2] < visual_thresh or skeletons[j][edge[
                    1], 2] < visual_thresh:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length / 2), stickwidth),
                                       int(angle), 0, 360, 1)
            if ids is None:
                color = colors[i] if color_set is None else colors[color_set[j]
                                                                   %
                                                                   len(colors)]
            else:
                color = get_color(ids[j])
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


@register
class KptOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(KptOutput, self).__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        for res in inputs:
            fn, image, keypoints, kpt_scores = res.values()
            res.pop('input.image')
            image = draw_kpt(image, keypoints)
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            logger.info(res)
            if self.save_img:
                file_name = os.path.split(fn)[-1]
                out_path = os.path.join(self.output_dir, file_name)
                logger.info('Save output image to {}'.format(out_path))
                cv2.imwrite(out_path, image)
            if self.save_res or self.return_res:
                total_res.append(res)
        if self.save_res:
            res_file_name = 'kpt_output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            logger.info('Save output result to {}'.format(out_path))
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
        if self.return_res:
            return total_res
        return
