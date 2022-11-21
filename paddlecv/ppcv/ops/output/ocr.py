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
import cv2
import json
import random
import math
from PIL import Image, ImageDraw, ImageFont
from .base import OutputBaseOp
from ppcv.utils.logger import setup_logger
from ppcv.core.workspace import register
from ppcv.utils.download import get_font_path

logger = setup_logger('OCROutput')


def draw_boxes(img, boxes):
    boxes = np.array(boxes)
    img_show = img.copy()
    for box in boxes.astype(int):
        if len(box) == 4:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
            cv2.polylines(img_show, [box], True, (0, 0, 255), 2)
    return img_show


@register
class OCRTableOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(OCRTableOutput, self).__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        for input in inputs:
            fn, image, dt_bboxes, structures, scores = list(input.values())[:5]
            res = dict(
                filename=fn,
                dt_bboxes=dt_bboxes,
                structures=structures,
                scores=scores)
            if 'Matcher.html' in input:
                res.update(html=input['Matcher.html'])
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            logger.info(res)
            if self.save_img:
                image = draw_boxes(image[:, :, ::-1], dt_bboxes)
                file_name = os.path.split(fn)[-1]
                out_path = os.path.join(self.output_dir, file_name)
                logger.info('Save output image to {}'.format(out_path))
                cv2.imwrite(out_path, image)
            if self.save_res or self.return_res:
                total_res.append(res)
        if self.save_res:
            res_file_name = 'output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            logger.info('Save output result to {}'.format(out_path))
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
        if self.return_res:
            return total_res
        return


@register
class OCROutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(OCROutput, self).__init__(model_cfg, env_cfg)
        font_path = model_cfg.get('font_path', None)
        self.font_path = get_font_path(font_path)

    def __call__(self, inputs):
        total_res = []
        for input in inputs:
            fn, image, dt_polys = list(input.values())[:3]
            rec_text = input.get('rec.rec_text', None)
            rec_score = input.get('rec.rec_score', None)
            res = dict(
                filename=fn,
                dt_polys=dt_polys.tolist(),
                rec_text=rec_text,
                rec_score=rec_score)
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            logger.info(res)
            if self.save_img:
                image = image[:, :, ::-1]
                if rec_text is not None:
                    image = self.draw_ocr_box_txt(
                        Image.fromarray(image), dt_polys, rec_text, rec_score)
                else:
                    image = draw_boxes(image, dt_polys.reshape([-1, 8]))
                file_name = os.path.split(fn)[-1]
                out_path = os.path.join(self.output_dir, file_name)
                logger.info('Save output image to {}'.format(out_path))
                cv2.imwrite(out_path, image)
            if self.save_res or self.return_res:
                total_res.append(res)
        if self.save_res:
            res_file_name = 'output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            logger.info('Save output result to {}'.format(out_path))
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
        if self.return_res:
            return total_res
        return

    def draw_ocr_box_txt(self,
                         image,
                         boxes,
                         txts=None,
                         scores=None,
                         drop_score=0.5):
        h, w = image.height, image.width
        img_left = image.copy()
        img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
        random.seed(0)

        draw_left = ImageDraw.Draw(img_left)
        if txts is None or len(txts) != len(boxes):
            txts = [None] * len(boxes)
        for idx, (box, txt) in enumerate(zip(boxes, txts)):
            if scores is not None and scores[idx] < drop_score:
                continue
            color = (random.randint(0, 255), random.randint(0, 255),
                     random.randint(0, 255))
            draw_left.polygon(box, fill=color)
            img_right_text = self.draw_box_txt_fine((w, h), box, txt)
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_right_text, [pts], True, color, 1)
            img_right = cv2.bitwise_and(img_right, img_right_text)
        img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
        return np.array(img_show)

    def draw_box_txt_fine(self, img_size, box, txt):
        box_height = int(
            math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2))
        box_width = int(
            math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2))

        if box_height > 2 * box_width and box_height > 30:
            img_text = Image.new('RGB', (box_height, box_width),
                                 (255, 255, 255))
            draw_text = ImageDraw.Draw(img_text)
            if txt:
                font = self.create_font(txt, (box_height, box_width))
                draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
            img_text = img_text.transpose(Image.ROTATE_270)
        else:
            img_text = Image.new('RGB', (box_width, box_height),
                                 (255, 255, 255))
            draw_text = ImageDraw.Draw(img_text)
            if txt:
                font = self.create_font(txt, (box_width, box_height))
                draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

        pts1 = np.float32(
            [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
        pts2 = np.array(box, dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)

        img_text = np.array(img_text, dtype=np.uint8)
        img_right_text = cv2.warpPerspective(
            img_text,
            M,
            img_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255))
        return img_right_text

    def create_font(self, txt, sz):
        font_size = int(sz[1] * 0.99)
        font = ImageFont.truetype(self.font_path, font_size, encoding="utf-8")
        length = font.getsize(txt)[0]
        if length > sz[0]:
            font_size = int(font_size * sz[0] / length)
            font = ImageFont.truetype(
                self.font_path, font_size, encoding="utf-8")
        return font


@register
class PPStructureOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(PPStructureOutput, self).__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        for res in inputs:
            image = res.pop(self.input_keys[1])
            res['concat.dt_polys'] = [
                x.tolist() for x in res['concat.dt_polys']
            ]
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            logger.info(res)
            if self.save_res or self.return_res:
                total_res.append(res)
        if self.save_res:
            res_file_name = 'output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            logger.info('Save output result to {}'.format(out_path))
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
        if self.return_res:
            return total_res
        return


@register
class PPStructureSerOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(PPStructureSerOutput, self).__init__(model_cfg, env_cfg)
        font_path = model_cfg.get('font_path', None)
        self.font_path = get_font_path(font_path)

    def __call__(self, inputs):
        total_res = []
        for input in inputs:
            fn, image = list(input.values())[:2]
            pred_ids = input.get(self.input_keys[2], None)
            preds = input.get(self.input_keys[3], None)
            dt_polys = input.get(self.input_keys[4], None)
            rec_texts = input.get(self.input_keys[5], None)
            res = dict(
                filename=fn,
                dt_polys=np.array(dt_polys).tolist(),
                rec_text=rec_texts,
                preds=preds,
                pred_ids=pred_ids)
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            logger.info(res)
            if self.save_img:
                image = self.draw_ser_results(image, pred_ids, preds, dt_polys,
                                              rec_texts)
                file_name = os.path.split(fn)[-1]
                out_path = os.path.join(self.output_dir, file_name)
                logger.info('Save output image to {}'.format(out_path))
                cv2.imwrite(out_path, image)
            if self.save_res or self.return_res:
                total_res.append(res)
        if self.save_res:
            res_file_name = 'output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            logger.info('Save output result to {}'.format(out_path))
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
        if self.return_res:
            return total_res
        return

    def draw_ser_results(self, image, pred_ids, preds, dt_polys, rec_texts):
        np.random.seed(2021)
        color = (np.random.permutation(range(255)),
                 np.random.permutation(range(255)),
                 np.random.permutation(range(255)))
        color_map = {
            idx: (color[0][idx], color[1][idx], color[2][idx])
            for idx in range(1, 255)
        }
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str) and os.path.isfile(image):
            image = Image.open(image).convert('RGB')
        img_new = image.copy()
        draw = ImageDraw.Draw(img_new)

        font = ImageFont.truetype(self.font_path, 14, encoding="utf-8")
        for pred_id, pred, dt_poly, rec_text in zip(pred_ids, preds, dt_polys,
                                                    rec_texts):
            if pred_id not in color_map:
                continue
            color = color_map[pred_id]
            text = "{}: {}".format(pred, rec_text)

            bbox = self.trans_poly_to_bbox(dt_poly)
            self.draw_box_txt(bbox, text, draw, font, 14, color)

        img_new = Image.blend(image, img_new, 0.7)
        return np.array(img_new)

    def draw_box_txt(self, bbox, text, draw, font, font_size, color):

        # draw ocr results outline
        bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
        draw.rectangle(bbox, fill=color)

        # draw ocr results
        tw = font.getsize(text)[0]
        th = font.getsize(text)[1]
        start_y = max(0, bbox[0][1] - th)
        draw.rectangle(
            [(bbox[0][0] + 1, start_y), (bbox[0][0] + tw + 1, start_y + th)],
            fill=(0, 0, 255))
        draw.text(
            (bbox[0][0] + 1, start_y), text, fill=(255, 255, 255), font=font)

    def trans_poly_to_bbox(self, poly):
        x1 = np.min([p[0] for p in poly])
        x2 = np.max([p[0] for p in poly])
        y1 = np.min([p[1] for p in poly])
        y2 = np.max([p[1] for p in poly])
        return [x1, y1, x2, y2]


@register
class PPStructureReOutput(PPStructureSerOutput):
    def __init__(self, model_cfg, env_cfg):
        super(PPStructureReOutput, self).__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        for input in inputs:
            fn, image = list(input.values())[:2]
            heads = input.get(self.input_keys[2], None)
            tails = input.get(self.input_keys[3], None)
            res = dict(filename=fn, heads=heads, tails=tails)
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            logger.info(res)
            if self.save_img:
                image = self.draw_re_results(image, heads, tails)
                file_name = os.path.split(fn)[-1]
                out_path = os.path.join(self.output_dir, file_name)
                logger.info('Save output image to {}'.format(out_path))
                cv2.imwrite(out_path, image)
            if self.save_res or self.return_res:
                total_res.append(res)
        if self.save_res:
            res_file_name = 'output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            logger.info('Save output result to {}'.format(out_path))
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
        if self.return_res:
            return total_res
        return

    def draw_re_results(self, image, heads, tails):
        font_size = 18
        np.random.seed(0)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str) and os.path.isfile(image):
            image = Image.open(image).convert('RGB')
        img_new = image.copy()
        draw = ImageDraw.Draw(img_new)

        font = ImageFont.truetype(self.font_path, font_size, encoding="utf-8")
        color_head = (0, 0, 255)
        color_tail = (255, 0, 0)
        color_line = (0, 255, 0)

        for ocr_info_head, ocr_info_tail in zip(heads, tails):
            head_bbox = self.trans_poly_to_bbox(ocr_info_head["dt_polys"])
            tail_bbox = self.trans_poly_to_bbox(ocr_info_tail["dt_polys"])
            self.draw_box_txt(head_bbox, ocr_info_head["rec_text"], draw, font,
                              font_size, color_head)
            self.draw_box_txt(tail_bbox, ocr_info_tail["rec_text"], draw, font,
                              font_size, color_tail)

            center_head = ((head_bbox[0] + head_bbox[2]) // 2,
                           (head_bbox[1] + head_bbox[3]) // 2)
            center_tail = ((tail_bbox[0] + tail_bbox[2]) // 2,
                           (tail_bbox[1] + tail_bbox[3]) // 2)

            draw.line([center_head, center_tail], fill=color_line, width=5)

        img_new = Image.blend(image, img_new, 0.5)
        return np.array(img_new)
