#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from .CGAN import CGAN
from .DCGAN import DCGAN
from .CycleGAN import CycleGAN
from .Pix2pix import Pix2pix
from .STGAN import STGAN
from .StarGAN import StarGAN
from .AttGAN import AttGAN
from .SPADE import SPADE

import importlib


def get_special_cfg(model_net):
    model = "trainer." + model_net
    modellib = importlib.import_module(model)
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_net.lower():
            model = cls()

    return model.add_special_args
