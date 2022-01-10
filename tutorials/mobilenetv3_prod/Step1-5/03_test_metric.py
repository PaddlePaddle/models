import torch
import paddle
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

from mobilenetv3_paddle.paddlevision.models import mobilenet_v3_small as mv3_small_paddle
from mobilenetv3_ref.torchvision.models import mobilenet_v3_small as mv3_small_torch
from mobilenetv3_ref import accuracy_torch
from mobilenetv3_paddle import accuracy_paddle
from utilities import build_paddle_data_pipeline, build_torch_data_pipeline
from utilities import evaluate


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
    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline()
    torch_dataset, torch_dataloader = build_torch_data_pipeline()
    for idx, (paddle_batch, torch_batch
              ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx > 0:
            break
        evaluate(paddle_batch[0], paddle_batch[1], paddle_model,
                 accuracy_paddle, 'paddle', reprod_logger)
        evaluate(torch_batch[0], torch_batch[1], torch_model, accuracy_torch,
                 'ref', reprod_logger)


if __name__ == "__main__":
    test_forward()

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/metric_ref.npy")
    paddle_info = diff_helper.load_info("./result/metric_paddle.npy")
    print(torch_info, paddle_info)

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/metric_diff.log")
