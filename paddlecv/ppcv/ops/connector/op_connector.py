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

__all__ = [
    "ClsCorrectionOp", "BboxCropOp", "PolyCropOp", "FragmentCompositionOp",
    "KeyFrameExtractionOp", "TableMatcherOp", 'TrackerOP', 'BboxExpandCropOp'
]

import cv2
import numpy as np
import importlib
from collections import defaultdict

from ppcv.core.workspace import register
from ppcv.ops.base import create_operators
from .base import ConnectorBaseOp
from .keyframes_extract_helper import LUVAbsDiffKeyFrameExtractor
from .table_matcher import TableMatcher
from .tracker import OCSORTTracker, ParserTrackerResults


@register
class ClsCorrectionOp(ConnectorBaseOp):
    """
    rotate
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.class_num = model_cfg["class_num"]
        assert self.class_num in [
            2, 4
        ], f"just [2, 4] are supported but got {self.class_num}"
        if self.class_num == 2:
            self.rotate_code = {1: cv2.ROTATE_180, }
        else:
            self.rotate_code = {
                1: cv2.ROTATE_90_COUNTERCLOCKWISE,
                2: cv2.ROTATE_180,
                3: cv2.ROTATE_90_CLOCKWISE,
            }

        self.threshold = model_cfg["threshold"]
        self.check_input_keys()
        return

    @classmethod
    def get_output_keys(self):
        return ["corr_image"]

    def check_input_keys(self, ):
        # image, cls_id, prob is needed.
        assert len(
            self.input_keys
        ) == 3, f"input key of {self} must be 3 but got {len(self.input_keys)}"

    def __call__(self, inputs):
        outputs = []
        for idx in range(len(inputs)):
            images = inputs[idx][self.input_keys[0]]
            cls_ids = inputs[idx][self.input_keys[1]]
            probs = inputs[idx][self.input_keys[2]]
            is_image_list = isinstance(images, (list, tuple))
            if is_image_list is not True:
                images = [images]
                cls_ids = [cls_ids]
                probs = [probs]
            output = []
            for image, cls_id, prob in zip(images, cls_ids, probs):
                cls_id = cls_id[0]
                prob = prob[0]
                corr_image = image.copy()
                if prob >= self.threshold and cls_id in self.rotate_code:
                    corr_image = cv2.rotate(corr_image,
                                            self.rotate_code[cls_id])
                output.append(corr_image)

            if is_image_list is not True:
                output = output[0]
            outputs.append(output)
        return outputs


@register
class BboxCropOp(ConnectorBaseOp):
    """
    BboxCropOp
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.check_input_keys()
        return

    @classmethod
    def get_output_keys(self):
        return ["crop_image"]

    def check_input_keys(self, ):
        # image, bbox is needed.
        assert len(
            self.input_keys
        ) == 2, f"input key of {self} must be 2 but got {len(self.input_keys)}"

    def __call__(self, inputs):
        outputs = []
        for idx in range(len(inputs)):
            images = inputs[idx][self.input_keys[0]]
            bboxes = inputs[idx][self.input_keys[1]]
            is_image_list = isinstance(images, (list, tuple))
            if is_image_list is not True:
                images = [images]
                bboxes = [bboxes]
            output = []
            # bbox: N x 4, x1, y1, x2, y2
            for image, bbox, in zip(images, bboxes):
                crop_imgs = []
                for single_bbox in np.array(bbox):
                    xmin, ymin, xmax, ymax = single_bbox.astype("int")
                    crop_img = image[ymin:ymax, xmin:xmax, :].copy()
                    crop_imgs.append(crop_img)
                output.append(crop_imgs)

            if is_image_list is not True:
                output = output[0]
            outputs.append({self.output_keys[0]: output})
        return outputs


@register
class PolyCropOp(ConnectorBaseOp):
    """
    PolyCropOp
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.check_input_keys()
        return

    @classmethod
    def get_output_keys(self):
        return ["crop_image"]

    def check_input_keys(self, ):
        # image, bbox is needed.
        assert len(
            self.input_keys
        ) == 2, f"input key of {self} must be 2 but got {len(self.input_keys)}"

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, 0, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def __call__(self, inputs):
        outputs = []
        for idx in range(len(inputs)):
            images = inputs[idx][self.input_keys[0]]
            polys = inputs[idx][self.input_keys[1]]
            is_image_list = isinstance(images, (list, tuple))
            if is_image_list is not True:
                images = [images]
                polys = [polys]
            output = []
            # bbox: N x 4 x 2, x1,y1, x2,y2, x3,y3, x4,y4
            for image, poly, in zip(images, polys):
                crop_imgs = []
                for single_poly in poly:
                    crop_img = self.get_rotate_crop_image(image, single_poly)
                    crop_imgs.append(crop_img)
                output.append(crop_imgs)

            if is_image_list is not True:
                output = output[0]

            outputs.append({self.output_keys[0]: output, })
        return outputs


@register
class FragmentCompositionOp(ConnectorBaseOp):
    """
    FragmentCompositionOp
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.split = model_cfg.get("split", " ")
        self.check_input_keys()
        return

    @classmethod
    def get_output_keys(self):
        return ["merged_text"]

    def check_input_keys(self, ):
        # list of string is needed
        assert len(
            self.input_keys
        ) == 1, f"input key of {self} must be 1 but got {len(self.input_keys)}"

    def __call__(self, inputs):
        outputs = []
        for idx in range(len(inputs)):
            strs = inputs[idx][self.input_keys[0]]
            output = self.split.join(strs)
            outputs.append(output)
        return outputs


@register
class KeyFrameExtractionOp(ConnectorBaseOp):
    """
    KeyFrameExtractionOp
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.check_input_keys()
        assert model_cfg["algo"] in ["luv_diff", ]
        if model_cfg["algo"] == "luv_diff":
            self.extractor = LUVAbsDiffKeyFrameExtractor(model_cfg["params"])

    @classmethod
    def get_output_keys(self):
        return ["key_frames", "key_frames_id"]

    def check_input_keys(self, ):
        # video is needed
        assert len(
            self.input_keys
        ) == 1, f"input key of {self} must be 1 but got {len(self.input_keys)}"

    def __call__(self, inputs):
        outputs = []
        for idx in range(len(inputs)):
            input = inputs[idx][self.input_keys[0]]
            key_frames, key_frames_id = self.extractor(input)
            outputs.append([key_frames, key_frames_id])
        return outputs


@register
class TableMatcherOp(ConnectorBaseOp):
    """
    TableMatcherOp
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.check_input_keys()
        filter_ocr_result = model_cfg.get("filter_ocr_result", False)
        self.matcher = TableMatcher(filter_ocr_result=filter_ocr_result)

    @classmethod
    def get_output_keys(self):
        return ["html"]

    def check_input_keys(self, ):
        #  pred_structure, pred_bboxes, dt_boxes, res_res are needed
        assert len(
            self.input_keys
        ) == 4, f"input key of {self} must be 4 but got {len(self.input_keys)}"

    def __call__(self, inputs):
        outputs = []
        for idx in range(len(inputs)):
            structure_bboxes = inputs[idx][self.input_keys[0]]
            structure_strs = inputs[idx][self.input_keys[1]]
            dt_boxes = inputs[idx][self.input_keys[2]]
            rec_res = inputs[idx][self.input_keys[3]]

            if len(structure_strs) == 0:
                outputs.append({self.output_keys[0]: ['']})
                continue
            is_list = isinstance(structure_strs[0], (list, tuple))
            if is_list is not True:
                structure_strs = [structure_strs]
                structure_bboxes = [structure_bboxes]
                dt_boxes = [dt_boxes]
                rec_res = [rec_res]

            output = []
            for single_structure_strs, single_structure_bboxes, single_dt_boxes, single_rec_res, in zip(
                    structure_strs, structure_bboxes, dt_boxes, rec_res):
                pred_html = self.matcher(single_structure_strs,
                                         np.array(single_structure_bboxes),
                                         single_dt_boxes.reshape([-1, 8]),
                                         single_rec_res)
                pred_html = '<html><body><table>' + pred_html + '</table></body></html>'
                output.append({self.output_keys[0]: pred_html})
            if is_list is not True:
                output = output[0]
            else:
                d = defaultdict(list)
                for item in output:
                    for k in self.output_keys:
                        d[k].append(item[k])
                output = d
            outputs.append(output)
        return outputs


@register
class PPStructureFilterOp(ConnectorBaseOp):
    """
    PPStructureFilterOp
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.keep_keys = model_cfg.get("keep_keys", [])
        self.check_input_keys()
        return

    @classmethod
    def get_output_keys(self):
        return ["image", "dt_polys", "rec_text"]

    def check_input_keys(self, ):
        # list of string is needed
        assert len(
            self.input_keys
        ) == 4, f"input key of {self} must be 4 but got {len(self.input_keys)}"

    def __call__(self, inputs):
        outputs = []
        for idx, input in enumerate(inputs):
            images, dt_polys, rec_text = [], [], []
            for i in range(len(input[self.input_keys[0]])):
                if input[self.input_keys[0]][i] in self.keep_keys:
                    images.append(input[self.input_keys[1]][i])
                    dt_polys.append(input[self.input_keys[2]][i])
                    rec_text.append(input[self.input_keys[3]][i])
            outputs.append({
                self.output_keys[0]: images,
                self.output_keys[1]: dt_polys,
                self.output_keys[2]: rec_text,
            })
        return outputs


@register
class PPStructureResultConcatOp(ConnectorBaseOp):
    """
    PPStructureResultConcatOp
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.keep_keys = model_cfg.get("keep_keys", [])
        self.check_input_keys()
        return

    @classmethod
    def get_output_keys(self):
        return [
            "dt_polys", "rec_text", "dt_bboxes", "html", "cell_bbox",
            "structures"
        ]

    def check_input_keys(self, ):
        # list of string is needed
        assert len(
            self.input_keys
        ) == 8, f"input key of {self} must be 8 but got {len(self.input_keys)}"

    def __call__(self, inputs):
        outputs = []
        for idx, input in enumerate(inputs):
            dt_polys, rec_text = [], []
            structures, html, layout_dt_bboxes, table_dt_bboxes, table_dt_polys, table_rec_text, txts_dt_polys, txts_rec_text = input.values(
            )
            dt_polys.extend(txts_dt_polys)
            rec_text.extend(txts_rec_text)
            dt_polys.extend(table_dt_polys)
            rec_text.extend(table_rec_text)
            input_html = [''] * len(txts_dt_polys) + html
            input_structures = [[] for _ in range(len(txts_dt_polys))
                                ] + structures
            cell_bbox = [[]
                         for _ in range(len(txts_dt_polys))] + table_dt_bboxes
            outputs.append({
                self.output_keys[0]: dt_polys,
                self.output_keys[1]: rec_text,
                self.output_keys[2]: layout_dt_bboxes,
                self.output_keys[3]: input_html,
                self.output_keys[4]: cell_bbox,
                self.output_keys[5]: input_structures,
            })
        return outputs


@register
class OCRRotateOp(ConnectorBaseOp):
    """
    OCRRotateOp
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.thresh = model_cfg.get("thresh", 0)
        self.cv_rotate_code = model_cfg.get('rotate_map', {
            '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
            '180': cv2.ROTATE_180,
            '270': cv2.ROTATE_90_CLOCKWISE
        })
        self.check_input_keys()

    @classmethod
    def get_output_keys(self):
        return ["image"]

    def check_input_keys(self, ):
        # list of string is needed
        assert len(
            self.input_keys
        ) == 3, f"input key of {self} must be 3 but got {len(self.input_keys)}"

    def __call__(self, inputs):
        outputs = []
        for idx, input in enumerate(inputs):
            image = input[self.input_keys[0]]
            label_name = input[self.input_keys[1]][0]
            score = input[self.input_keys[2]][0]
            if score > self.thresh and label_name in self.cv_rotate_code:
                image = cv2.rotate(image, self.cv_rotate_code[label_name])
            outputs.append({self.output_keys[0]: image, })
        return outputs


@register
class TrackerOP(ConnectorBaseOp):
    """
    tracker
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.tracker_type = model_cfg['type']
        assert self.tracker_type in ['OCSORTTracker'
                                     ], f"Only OCSORTTracker is supported now"
        tracker_kwargs = model_cfg['tracker_configs']
        self.tracker = eval(self.tracker_type)(**tracker_kwargs)
        self.check_input_keys()
        mod = importlib.import_module(__name__)
        self.postprocessor = create_operators(model_cfg["PostProcess"], mod)

    @classmethod
    def get_output_keys(self):
        return [
            "tk_bboxes", "tk_scores", "tk_ids", "tk_cls_ids", "tk_cls_names"
        ]

    def check_input_keys(self):
        # "dt_bboxes", "dt_scores", "dt_class_ids" or plus reid feature
        assert len(self.input_keys) in [
            3, 4
        ], 'for OCSORTTracker, now only supported det ouputs and reid outputs'

    def create_inputs(self, det_result):
        dt_bboxes = np.array(det_result[self.input_keys[0]])
        dt_scores = np.array(det_result[self.input_keys[1]])
        dt_class_ids = np.array(det_result[self.input_keys[2]])
        dt_preds = np.concatenate(
            [dt_class_ids[:, None], dt_scores[:, None], dt_bboxes], axis=-1)
        if len(self.input_keys) > 3:
            dt_embs = np.array(det_result[self.input_keys[3]])
        else:
            dt_embs = None

        return dt_preds, dt_embs

    def postprocess(self, outputs):
        for idx, ops in enumerate(self.postprocessor):
            if idx == len(self.postprocessor) - 1:
                output_keys = ops(outputs, self.output_keys)
            else:
                outputs = ops(outputs)
        return outputs

    def __call__(self, inputs):
        pipe_outputs = []
        for input in inputs:
            dt_preds, dt_embs = self.create_inputs(input)
            tracking_outs = self.tracker.tracking(dt_preds, dt_embs,
                                                  self.output_keys)
            tracking_outs = self.postprocess(tracking_outs)
            pipe_outputs.append(tracking_outs)

        return pipe_outputs


@register
class BboxExpandCropOp(ConnectorBaseOp):
    """
    BboxExpandCropOp
    """

    def __init__(self, model_cfg, env_cfg=None):
        super().__init__(model_cfg, env_cfg)
        self.expand_ratio = model_cfg.get('expand_ratio', 0.3)
        self.check_input_keys()

    @classmethod
    def get_output_keys(self):
        return ['crop_image', 'tl_point']

    def check_input_keys(self, ):
        # image, bbox is needed.
        assert len(
            self.input_keys
        ) == 2, f"input key of {self} must be 2 but got {len(self.input_keys)}"

    def expand_crop(self, image, box):
        imgh, imgw, c = image.shape
        xmin, ymin, xmax, ymax = [int(x) for x in box]
        h_half = (ymax - ymin) * (1 + self.expand_ratio) / 2.
        w_half = (xmax - xmin) * (1 + self.expand_ratio) / 2.
        if h_half > w_half * 4 / 3:
            w_half = h_half * 0.75
        center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
        ymin = max(0, int(center[0] - h_half))
        ymax = min(imgh - 1, int(center[0] + h_half))
        xmin = max(0, int(center[1] - w_half))
        xmax = min(imgw - 1, int(center[1] + w_half))
        return image[ymin:ymax, xmin:xmax, :], [xmin, ymin]

    def __call__(self, inputs):
        outputs = []
        for idx in range(len(inputs)):
            images = inputs[idx][self.input_keys[0]]
            bboxes = inputs[idx][self.input_keys[1]]
            is_image_list = isinstance(images, (list, tuple))
            if is_image_list is False:
                images = [images]
                bboxes = [bboxes]

            output_images = []
            output_points = []
            # bbox: N x 4, x1, y1, x2, y2
            for image, bbox, in zip(images, bboxes):
                crop_imgs = []
                tl_points = []
                for single_bbox in bbox:
                    crop_img, tl_point = self.expand_crop(image, single_bbox)
                    crop_imgs.append(crop_img)
                    tl_points.append(tl_point)
                output_images.append(crop_imgs)
                output_points.append(tl_points)

            if is_image_list is False:
                output_images = output_images[0]
                output_points = output_points[0]
            outputs.append({
                self.output_keys[0]: output_images,
                self.output_keys[1]: output_points
            })
        return outputs
