#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid


class DetectorBase(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # variable of data layers
        self.feed_vars = {}
        # PyReader object
        self.pyreader = None

    def train(self):
        raise NotImplementedError('%s.train not available.' %
                                  (self.__class__.__name__))

    def test(self):
        raise NotImplementedError('%s.test not available.' %
                                  (self.__class__.__name__))

    def build_feeds(self, feed_info, use_pyreader=True):
        var = []
        for info in feed_info:
            d = fluid.layers.data(
                name=info['name'],
                shape=info['shape'],
                dtype=info['dtype'],
                lod_level=info['lod_level'])
            var.append(d)
            self.feed_vars[info['name']] = d
        if use_pyreader:
            self.pyreader = fluid.io.PyReader(
                feed_list=var,
                capacity=64,
                use_double_buffer=True,
                iterable=False)
        return self.feed_vars

    def get_pyreader(self):
        """
        """
        if self.reader is None:
            raise ValueError("{}.pyreader is not initialized.".format(
                self.__class__.__name__))
