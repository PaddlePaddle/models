# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle


class RunConfig(object):
    """
    Running Config Setting.

    Args:
        save_dir (obj:`str`): The directory to save checkpoint during training.
        use_gpu (obj:`bool`, optinal, defaults to obj:`False`): Whether use GPU for training, input should be True or False.
        lr (obj:`float`, optinal, defaults to 5e-4): Learning rate used to train.
        batch_size (obj:`int`, optinal, defaults to 1): Total examples' number of a batch.
        epochs (obj:`int`, optinal, defaults to 1): Number of epoches for training.
        log_freq (obj:`int`, optinal, defaults to 10): The frequency, in number of steps, the training logs are printed.
        eval_freq (obj:`int`, optinal, defaults to 1): The frequency, in number of epochs, an evalutation is performed.
        save_freq (obj:`int`, optinal, defaults to 1): The frequency, in number of epochs, to save checkpoints.

    """

    def __init__(self,
                 save_dir,
                 use_gpu=0,
                 lr=5e-4,
                 batch_size=1,
                 epochs=1,
                 log_freq=10,
                 eval_freq=1,
                 save_freq=1):
        self.save_dir = save_dir
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.save_freq = save_freq

        self._place = paddle.set_device(
            "gpu") if self.use_gpu else paddle.set_device("cpu")

    def get_save_dir(self):
        return self.save_dir

    def get_running_place(self):
        return self._place
