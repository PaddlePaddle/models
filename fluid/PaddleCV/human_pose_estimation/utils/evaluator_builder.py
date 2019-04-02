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

"""Interface for building evaluator."""

from utils.coco_evaluator import COCOEvaluator
from utils.mpii_evaluator import MPIIEvaluator


evaluator_map = {
    'coco': COCOEvaluator,
    'mpii': MPIIEvaluator
}


def create_evaluator(dataset):
    """
    :param dataset: specific dataset to be evaluated
    :return:
    """
    return evaluator_map[dataset]
