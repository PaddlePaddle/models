# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from .dcase import TAUUrbanAcousticScenes_2020_Mobile_DevelopmentSet
from .esc50 import ESC50
from .gtzan import GTZAN
from .urban_sound import UrbanSound8K

__all__ = [
    'ESC50',
    'UrbanSound8K',
    'GTZAN',
    'TAUUrbanAcousticScenes_2020_Mobile_DevelopmentSet',
]
