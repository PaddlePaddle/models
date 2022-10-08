# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np

from .utils import np2torch, np2paddle, paddle2np, torch2np, check_print_diff


def check_data(data1: dict, data2: dict):
    for k in data1:
        if k not in data2:
            assert k in data2, 'k in data1 but not found in data2'.format(
                k, data2)

    for k in data2:
        if k not in data1:
            assert k in data1, 'k in data2 but not found in data1'.format(
                k, data2.keys())


def compute_diff(data1: dict, data2: dict):
    out_dict = {}
    for k in data1:
        assert k in data2
        sub_data1, sub_data2 = data1[k], data2[k]
        assert type(sub_data1) == type(sub_data2)
        if isinstance(sub_data1, dict):
            out = compute_diff(sub_data1, sub_data2)
            out_dict[k] = out
        elif isinstance(sub_data1, np.ndarray):
            if sub_data1.shape != sub_data2.shape and sub_data1.transpose(
            ).shape == sub_data2.shape:
                print('transpose sub_data1')
                sub_data1 = sub_data1.transpose()
            diff = np.abs(sub_data1 - sub_data2)
            out_dict[k] = {
                'mean': diff.mean(),
                'max': diff.max(),
                'min': diff.min()
            }
        else:
            raise NotImplementedError
    return out_dict


def compare_forward(torch_model,
                    paddle_model: paddle.nn.Layer,
                    input_dict: dict,
                    diff_threshold: float=1e-6,
                    diff_method: str='mean'):
    torch_input = np2torch(input_dict)
    paddle_input = np2paddle(input_dict)

    torch_model.eval()
    paddle_model.eval()
    torch_out = torch_model(**torch_input)
    paddle_out = paddle_model(**paddle_input)

    diff_dict = compute_diff(torch2np(torch_out), paddle2np(paddle_out))
    passed = check_print_diff(
        diff_dict,
        diff_method=diff_method,
        diff_threshold=diff_threshold,
        print_func=print)
    if passed:
        print('diff check passed')
    else:
        print('diff check failed')


def compare_loss_and_backward(torch_model,
                              paddle_model: paddle.nn.Layer,
                              torch_loss,
                              paddle_loss: paddle.nn.Layer,
                              input_dict: dict,
                              lr: float=1e-3,
                              steps: int=10,
                              diff_threshold: float=1e-6,
                              diff_method: str='mean'):
    import torch

    torch_input = np2torch(input_dict)
    paddle_input = np2paddle(input_dict)

    torch_model.eval()
    paddle_model.eval()

    torch_optim = torch.optim.SGD(params=torch_model.parameters(), lr=lr)
    paddle_optim = paddle.optimizer.SGD(parameters=paddle_model.parameters(),
                                        learning_rate=lr)

    for i in range(steps):
        # paddle
        paddle_outputs = paddle_model(**paddle_input)
        paddle_loss_value = paddle_loss(paddle_input, paddle_outputs)
        paddle_loss_value['loss'].backward()
        paddle_optim.step()

        paddle_grad_dict = {'loss': paddle_loss_value['loss'].numpy()}
        for name, parms in paddle_model.named_parameters():
            if not parms.stop_gradient and parms.grad is not None:
                paddle_grad_dict[name] = parms.grad.numpy()
        paddle_optim.clear_grad()

        # torch
        torch_outputs = torch_model(**torch_input)
        torch_loss_value = torch_loss(torch_input, torch_outputs)
        torch_loss_value['loss'].backward()
        torch_optim.step()

        torch_grad_dict = {'loss': torch_loss_value['loss'].detach().numpy()}
        for name, parms in torch_model.named_parameters():
            if parms.requires_grad and parms.grad is not None:
                torch_grad_dict[name] = parms.grad.numpy()
        torch_optim.zero_grad()

        # compare
        diff_dict = compute_diff(paddle_grad_dict, torch_grad_dict)
        passed = check_print_diff(
            diff_dict,
            diff_method=diff_method,
            diff_threshold=diff_threshold,
            print_func=print)
        if passed:
            print('diff check passed in iter {}'.format(i))
        else:
            print('diff check failed in iter {}'.format(i))
            return
    print('diff check passed')
