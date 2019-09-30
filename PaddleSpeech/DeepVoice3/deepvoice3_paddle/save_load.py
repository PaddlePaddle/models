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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import dirname, join

import paddle
from paddle import fluid
import paddle.fluid.dygraph as dg


def _load(checkpoint_path):
    """
    Load saved state dict and optimizer state(optional).
    """
    state_dict, optimizer_state = dg.load_persistables(dirname=checkpoint_path)
    return state_dict, optimizer_state


def load_checkpoint(path, model, optimizer=None, reset_optimizer=True):
    """
    layers like FC, Conv*, ... the Layer does not initialize their parameters 
    before first run.
    
    1. if you want to load only a part of a saved whole model, to part of an 
    existing model, just pass the part as the target model , and path of the 
    saved whole model as source path.
    2. if you want to load exactly from what is saved, just passed the model 
    and path as expected.

    The rule of thumb is:
    1. loading to a model works with name, a unique global name.
    2. loading from a directory works with file structure, each parameter is
    saved in a file. Loading a file from directory A/ would `create` a
    corresponding Variable for each saved parameter, whose name is the file's
    relative path from directory A/.
    """
    print("Load checkpoint from: {}".format(path))
    state_dict, optimizer_state = _load(path)

    model.load_dict(state_dict)
    if not reset_optimizer and optimizer is not None:
        if optimizer_state is not None:
            print("[loading] Load optimizer state from {}".format(path))
            optimizer.load(optimizer_state)

    return model


def _load_embedding(path, model):
    print("[loading] Loading embedding from {}".format(path))
    state_dict, optimizer_state = _load(path)
    key = os.path.join(model.full_name(), "ConvS2S_0/Encoder_0/Embedding_0.w_0")
    tensor = model.state_dict()[key]._ivar.value().get_tensor()
    tensor.set(state_dict[key], fluid.framework._current_expected_place())


def save_checkpoint(model, optimizer, checkpoint_dir, global_step):
    checkpoint_path = join(checkpoint_dir,
                           "checkpoint_step{:09d}.model".format(global_step))
    dg.save_persistables(
        model.state_dict(), dirname=checkpoint_path, optimizers=optimizer)
    print("[checkpoint] Saved checkpoint:", checkpoint_path)
