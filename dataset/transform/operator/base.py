# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# function:
#    operators to process sample, 
#    eg: decode/resize/crop image

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import uuid
import numpy as np
import cv2

class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ process a sample

        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing

        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return '%s' % (self._id)


class DecodeImage(BaseOperator):
    def __init__(self, to_rgb=True, channel_first=False):
        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.channel_first = channel_first

    def __call__(self, sample, context=None):
        assert 'image' in sample, 'not found image data'
        img = sample['image']

        data = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        sample['image'] = img
        return sample


class ResizeImage(BaseOperator):
    pass
