import paddle
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

from mobilenetv3_paddle.paddlevision.models import mobilenet_v3_small as mv3_small_paddle
from mobilenetv3_ref.torchvision.models import mobilenet_v3_small as mv3_small_torch


def train_one_epoch_paddle(inputs, labels, model, criterion, optimizer,
                           lr_scheduler, max_iter, reprod_logger):
    for idx in range(max_iter):
        image = paddle.to_tensor(inputs, dtype="float32")
        target = paddle.to_tensor(labels, dtype="int64")
        # import pdb; pdb.set_trace()

        output = model(image)
        loss = criterion(output, target)

        reprod_logger.add("loss_{}".format(idx), loss.cpu().detach().numpy())
        reprod_logger.add("lr_{}".format(idx), np.array(lr_scheduler.get_lr()))

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step() 

    reprod_logger.save("./result/losses_paddle.npy")


def train_one_epoch_torch(inputs, labels, model, criterion, optimizer,
                          lr_scheduler, max_iter, reprod_logger):
    for idx in range(max_iter):
        image = torch.tensor(inputs, dtype=torch.float32).cuda()
        target = torch.tensor(labels, dtype=torch.int64).cuda()
        model = model.cuda()

        output = model(image)
        loss = criterion(output, target)

        reprod_logger.add("loss_{}".format(idx), loss.cpu().detach().numpy())
        reprod_logger.add("lr_{}".format(idx),
                          np.array(lr_scheduler.get_last_lr()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

    reprod_logger.save("./result/losses_ref.npy")


def test_backward():
    max_iter = 3
    lr = 1e-3
    momentum = 0.9
    lr_gamma = 0.1

    # set determinnistic flag
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    FLAGS_cudnn_deterministic = True

    # load paddle model
    paddle.set_device("gpu")
    paddle_model = mv3_small_paddle(dropout=0.0)
    paddle_model.eval()
    paddle_state_dict = paddle.load("./data/mv3_small_paddle.pdparams")
    paddle_model.set_dict(paddle_state_dict)

    # load torch model
    torch_model = mv3_small_torch(dropout=0.0)
    torch_model.eval()
    torch_state_dict = torch.load("./data/mobilenet_v3_small-047dcff4.pth")
    torch_model.load_state_dict(torch_state_dict, strict=False)

    # init loss
    criterion_paddle = paddle.nn.CrossEntropyLoss()
    criterion_torch = torch.nn.CrossEntropyLoss()

    # init optimizer
    lr_scheduler_paddle = paddle.optimizer.lr.StepDecay(
        lr, step_size=max_iter // 3, gamma=lr_gamma)
    opt_paddle = paddle.optimizer.Momentum(
        learning_rate=lr,
        momentum=momentum,
        parameters=paddle_model.parameters())

    opt_torch = torch.optim.SGD(torch_model.parameters(),
                                lr=lr,
                                momentum=momentum)
    lr_scheduler_torch = lr_scheduler.StepLR(
        opt_torch, step_size=max_iter // 3, gamma=lr_gamma)

    # prepare logger & load data
    reprod_logger = ReprodLogger()
    inputs = np.load("./data/fake_data.npy")
    labels = np.load("./data/fake_label.npy")

    train_one_epoch_paddle(inputs, labels, paddle_model, criterion_paddle,
                           opt_paddle, lr_scheduler_paddle, max_iter,
                           reprod_logger)

    train_one_epoch_torch(inputs, labels, torch_model, criterion_torch,
                          opt_torch, lr_scheduler_torch, max_iter,
                          reprod_logger)


if __name__ == "__main__":
    test_backward()

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/losses_ref.npy")
    paddle_info = diff_helper.load_info("./result/losses_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/backward_diff.log")
