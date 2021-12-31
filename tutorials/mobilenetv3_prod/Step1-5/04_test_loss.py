# add loss comparing code

import torch
import paddle
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

from mobilenetv3_paddle.paddlevision.models import mobilenet_v3_small as mv3_small_paddle
from mobilenetv3_ref.torchvision.models import mobilenet_v3_small as mv3_small_torch


def test_forward():
    # init loss
    criterion_paddle = paddle.nn.CrossEntropyLoss()
    criterion_torch = torch.nn.CrossEntropyLoss()

    # load paddle model
    paddle_model = mv3_small_paddle()
    paddle_model.eval()
    paddle_state_dict = paddle.load("./data/mv3_small_paddle.pdparams")
    paddle_model.set_dict(paddle_state_dict)

    # load torch model
    torch_model = mv3_small_torch()
    torch_model.eval()
    torch_state_dict = torch.load("./data/mobilenet_v3_small-047dcff4.pth")
    torch_model.load_state_dict(torch_state_dict)

    # prepare logger & load data
    reprod_logger = ReprodLogger()
    inputs = np.load("./data/fake_data.npy")
    labels = np.load("./data/fake_label.npy")

    # save the paddle output
    paddle_out = paddle_model(paddle.to_tensor(inputs, dtype="float32"))
    loss_paddle = criterion_paddle(
        paddle_out, paddle.to_tensor(
            labels, dtype="int64"))
    reprod_logger.add("loss", loss_paddle.cpu().detach().numpy())
    reprod_logger.save("./result/loss_paddle.npy")

    # save the torch output
    torch_out = torch_model(torch.tensor(inputs, dtype=torch.float32))
    loss_torch = criterion_torch(
        torch_out, torch.tensor(
            labels, dtype=torch.int64))
    reprod_logger.add("loss", loss_torch.cpu().detach().numpy())
    reprod_logger.save("./result/loss_ref.npy")


if __name__ == "__main__":
    test_forward()

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/loss_ref.npy")
    paddle_info = diff_helper.load_info("./result/loss_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/loss_diff.log")
