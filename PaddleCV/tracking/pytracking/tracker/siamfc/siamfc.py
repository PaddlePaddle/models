import time
import numpy as np

import paddle.fluid as fluid
from paddle.fluid import dygraph

from pytracking.tracker.base.basetracker import BaseTracker

from ltr.models.siamese.siam import siamfc_alexnet

import cv2
# for debug
from pytracking.parameter.siamfc.default import parameters


class SiamFC(BaseTracker):
    def __init__(self, params=parameters()):

        self.params = params
        self.model_initializer()

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.features.initialize()
        self.features_initialized = True

    def model_initializer(self):
        import os
        net_path = self.params.net_path
        if net_path is None:
            net_path = self.params.features.features[0].net_path
        if not os.path.exists(net_path):
            raise Exception("not found {}".format(net_path))
        with dygraph.guard():
            self.model = siamfc_alexnet(backbone_is_test=True)
            #state_dict, _ = fluid.load_dygraph(net_path)
            weight_params, opt_params = fluid.load_dygraph(net_path)
            state_dict = self.model.state_dict()
            for k1, k2 in zip(state_dict.keys(), weight_params.keys()):
                if list(state_dict[k1].shape) == list(weight_params[k2].shape):
                    state_dict[k1].set_value(weight_params[k2])
                else:
                    raise Exception("ERROR, shape not match")
            self.model.load_dict(state_dict)
            self.model.eval()

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(
            np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def initialize(self, image, state, *args, **kwargs):
        # state (x, y, w, h)
        # Initialize some stuff
        self.frame_num = 1
        self.time = 0

        # Get position and size
        box = state
        image = np.asarray(image)
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array(
            [
                box[1] - 1 + (box[3] - 1) / 2, box[0] - 1 + (box[2] - 1) / 2,
                box[3], box[2]
            ],
            dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.params.response_up * self.params.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz), np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.params.scale_step**np.linspace(
            -(self.params.scale_num // 2), self.params.scale_num // 2,
            self.params.scale_num)

        # exemplar and search sizes
        context = self.params.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.params.instance_sz / self.params.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image,
            self.center,
            self.z_sz,
            out_size=self.params.exemplar_sz,
            pad_color=self.avg_color)
        self.exemplar_img_1s = exemplar_image[np.newaxis, :, :, :]
        self.exemplar_img = np.transpose(self.exemplar_img_1s,
                                         [0, 3, 1, 2]).astype(np.float32)
        self.exemplar_img = np.repeat(
            self.exemplar_img, self.params.scale_num, axis=0)

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((np.round(center - (size - 1) / 2),
                                  np.round(center - (size - 1) / 2) + size))
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

        return patch

    def track(self, image):
        #print("## track, input image shape:", image.shape)
        self.frame_num += 1

        image = np.asarray(image)
        # search images
        instance_images = [
            self._crop_and_resize(
                image,
                self.center,
                self.x_sz * f,
                out_size=self.params.instance_sz,
                pad_color=self.avg_color) for f in self.scale_factors
        ]
        instance_images = np.stack(instance_images, axis=0)
        instance_images = np.transpose(instance_images,
                                       [0, 3, 1, 2]).astype(np.float32)

        # calculate response
        # exemplar features
        with fluid.dygraph.guard():
            instance_images = fluid.dygraph.to_variable(instance_images)
            self.exemplar_img = fluid.dygraph.to_variable(self.exemplar_img)
            responses = self.model(self.exemplar_img, instance_images)

        responses = responses.numpy()

        responses = np.squeeze(responses, axis=1)
        # upsample responses and penalize scale changes
        responses = np.stack(
            [
                cv2.resize(
                    t, (self.upscale_sz, self.upscale_sz),
                    interpolation=cv2.INTER_CUBIC) for t in responses
            ],
            axis=0)
        responses[:self.params.scale_num // 2] *= self.params.scale_penalty
        responses[self.params.scale_num // 2 + 1:] *= self.params.scale_penalty

        # peak scale
        scale_list = np.amax(responses, axis=(1, 2))
        scale_id = np.argmax(scale_list)
        #scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.params.window_influence) * response + \
                   self.params.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1.) / 2
        disp_in_instance = disp_in_response * \
                           self.params.total_stride / self.params.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / self.params.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - self.params.scale_lr) * 1.0 + \
                self.params.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2, self.target_sz[1],
            self.target_sz[0]
        ])

        return box
