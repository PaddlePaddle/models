# add test metric code paddle vs torch

import torch
import paddle
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

from mobilenetv3_paddle.paddlevision.models import mobilenet_v3_small as mv3_small_paddle
from mobilenetv3_ref.torchvision.models import mobilenet_v3_small as mv3_small_torch
from mobilenetv3_ref import accuracy_torch
from mobilenetv3_paddle import accuracy_paddle


def evaluate(image, labels, model, acc, tag, reprod_logger):
    model.eval()
    output = model(image)

    accracy = acc(output, labels, topk=(1, 5))

    reprod_logger.add("acc_top1", np.array(accracy[0]))
    reprod_logger.add("acc_top5", np.array(accracy[1]))

    reprod_logger.save("./result/metric_{}.npy".format(tag))


def test_forward():
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
    image = paddle.to_tensor(inputs, dtype="float32")
    target = paddle.to_tensor(labels, dtype="int64")

    evaluate(
        paddle.to_tensor(
            inputs, dtype="float32"),
        paddle.to_tensor(
            labels, dtype="int64"),
        paddle_model,
        accuracy_paddle,
        'paddle',
        reprod_logger)
    evaluate(
        torch.tensor(
            inputs, dtype=torch.float32),
        torch.tensor(
            labels, dtype=torch.int64),
        torch_model,
        accuracy_torch,
        'ref',
        reprod_logger)


if __name__ == "__main__":
    test_forward()

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/metric_ref.npy")
    paddle_info = diff_helper.load_info("./result/metric_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/metric_diff.log")
