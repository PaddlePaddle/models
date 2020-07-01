import time
import math
import cv2
import numpy as np

import paddle.fluid as fluid
from paddle.fluid import dygraph

from pytracking.tracker.base.basetracker import BaseTracker
from ltr.data.anchor import Anchors


class SiamMask(BaseTracker):
    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.features.initialize()
        self.features_initialized = True

    def initialize(self, image, state, *args, **kwargs):
        # Initialize some stuff
        self.frame_num = 1

        # Initialize features
        self.initialize_features()

        self.time = 0
        tic = time.time()

        # Get position and size
        # self.pos: target center (y, x)
        self.pos = np.array(
            [
                state[1] + state[3] // 2,
                state[0] + state[2] // 2
            ],
            dtype=np.float32)
        self.target_sz = np.array([state[3], state[2]], dtype=np.float32)

        # Set search area
        context = self.params.context_amount * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = round(self.z_sz * (self.params.instance_size / self.params.exemplar_size))

        self.score_size = (self.params.instance_size - self.params.exemplar_size) // \
            self.params.anchor_stride + 1 + self.params.base_size
        self.anchor_num = len(self.params.anchor_ratios) * len(self.params.anchor_scales)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)

        # Convert image
        self.avg_color = np.mean(image, axis=(0, 1))
        with dygraph.guard():
            exemplar_image = self._crop_and_resize(
                image,
                self.pos,
                self.z_sz,
                out_size=self.params.exemplar_size,
                pad_color=self.avg_color)
            
            # get template
            self.params.features.features[0].net.template(exemplar_image)

        self.time += time.time() - tic
    
    def track(self, image):
        self.frame_num += 1

        # Convert image
        image = np.asarray(image)

        with dygraph.guard():
            # search images
            instance_image = self._crop_and_resize(
                image,
                self.pos,
                self.x_sz,
                out_size=self.params.instance_size,
                pad_color=self.avg_color)
            instance_box = [
                self.pos[1] - self.x_sz / 2,
                self.pos[0] - self.x_sz / 2,
                self.x_sz,
                self.x_sz]
            # predict
            output = self.params.features.features[0].net.track(instance_image)
            score = self._convert_score(output['cls'])
            pred_bbox = self._convert_bbox(output['loc'], self.anchors)
        
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        scale_z = self.params.exemplar_size / self.z_sz
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.target_sz[1]*scale_z, self.target_sz[0]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.target_sz[1]/self.target_sz[0]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.params.penalty_k)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.params.window_influence) + \
            self.window * self.params.window_influence
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.params.lr

        cx = bbox[0] + self.pos[1]
        cy = bbox[1] + self.pos[0]

        # smooth bbox
        width = self.target_sz[1] * (1 - lr) + bbox[2] * lr
        height = self.target_sz[0] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, image.shape[:2])
        
        # update state
        self.pos = np.array([cy, cx])
        self.target_sz = np.array([height, width])
        context = self.params.context_amount * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = round(self.z_sz * (self.params.instance_size / self.params.exemplar_size))

        if self.params.features.features[0].net.refine_head is None or not self.params.polygon:
            # Return new state
            yx = self.pos - self.target_sz / 2
            new_state = np.array([yx[1], yx[0], self.target_sz[1], self.target_sz[0]], 'float32')
            return new_state.tolist()

        # processing mask
        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = int(pos[2]), int(pos[1])
        with dygraph.guard():
            mask = self.params.features.features[0].net.mask_refine((delta_y, delta_x))
            mask = fluid.layers.sigmoid(mask)
            mask = fluid.layers.reshape(mask, [-1])
            out_size = self.params.mask_output_size
            mask = fluid.layers.reshape(mask,[out_size, out_size]).numpy()

        s = instance_box[2] / self.params.instance_size
        base_size = self.params.base_size
        stride = self.params.anchor_stride
        sub_box = [instance_box[0] + (delta_x - base_size/2) * stride * s,
                   instance_box[1] + (delta_y - base_size/2) * stride * s,
                   s * self.params.exemplar_size,
                   s * self.params.exemplar_size]
        s = out_size / sub_box[2]

        im_h, im_w = image.shape[:2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        polygon = self._mask_post_processing(mask_in_img)
        # Return new state
        new_state = polygon.flatten()

        return new_state.tolist()

    def generate_anchor(self, score_size):
        anchors = Anchors(
            self.params.anchor_stride,
            self.params.anchor_ratios,
            self.params.anchor_scales)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid(
            [ori + total_stride * dx for dx in range(score_size)],
            [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate(
            (
                np.floor(center - (size + 1) / 2 + 0.5),
                np.floor(center - (size + 1) / 2 + 0.5) + size
            ))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((-corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image,
                npad,
                npad,
                npad,
                npad,
                cv2.BORDER_CONSTANT,
                value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        patch = patch.transpose(2, 0, 1)
        patch = patch[np.newaxis, :, :, :]
        patch = patch.astype(np.float32)
        patch = fluid.dygraph.to_variable(patch)
        return patch

    def _convert_bbox(self, delta, anchor):
        delta = fluid.layers.transpose(delta, [1, 2, 3, 0])
        delta = fluid.layers.reshape(delta, [4, -1]).numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = fluid.layers.transpose(score, [1, 2, 3, 0])
        score = fluid.layers.reshape(score, [2, -1])
        score = fluid.layers.transpose(score, [1, 0])
        score = fluid.layers.softmax(score, axis=1)[:, 1].numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(
            image,
            mapping,
            (out_sz[0], out_sz[1]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > self.params.mask_threshold)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(
                target_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(
                target_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
            rbox_in_img = prbox
        else:  # empty mask
            yx = self.pos - self.target_sz / 2
            location = np.array([yx[1], yx[0], self.target_sz[1], self.target_sz[0]], 'float32')
            rbox_in_img = np.array(
                [
                    [location[0], location[1]],
                    [location[0] + location[2], location[1]],
                    [location[0] + location[2], location[1] + location[3]],
                    [location[0], location[1] + location[3]]
                ])
        return rbox_in_img

