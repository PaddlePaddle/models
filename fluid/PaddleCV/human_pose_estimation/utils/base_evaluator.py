# Copyright (c) 2019-present, Baidu, Inc.
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
##############################################################################

"""Interface for evaluation."""


class BaseEvaluator(object):
    def __init__(self, root, kp_dim):
        """
        :param root: the root dir of dataset
        :param kp_dim: the dimension of keypoints
        """
        self.root = root
        self.kp_dim = kp_dim

    def evaluate(self, *args, **kwargs):
        """
        Need Implementation for specific task / dataset
        """
        raise NotImplementedError
