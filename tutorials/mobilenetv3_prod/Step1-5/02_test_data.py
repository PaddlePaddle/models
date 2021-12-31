import os
import sys
import torch
import paddle
import numpy as np
from PIL import Image
from reprod_log import ReprodLogger, ReprodDiffHelper
import mobilenetv3_paddle.presets as presets_paddle
import mobilenetv3_paddle.paddlevision as paddlevision
import mobilenetv3_ref.presets as presets_torch
import mobilenetv3_ref.torchvision as torchvision


def build_paddle_data_pipeline():
    # dataset & data_loader
    dataset_test = paddlevision.datasets.ImageFolder(
        "./lite_data/val/",
        presets_paddle.ClassificationPresetEval(
            crop_size=224, resize_size=256))

    test_sampler = paddle.io.SequenceSampler(dataset_test)

    test_batch_sampler = paddle.io.BatchSampler(
        sampler=test_sampler, batch_size=4)

    data_loader_test = paddle.io.DataLoader(
        dataset_test, batch_sampler=test_batch_sampler, num_workers=0)

    return dataset_test, data_loader_test


def build_torch_data_pipeline():
    dataset_test = torchvision.datasets.ImageFolder(
        "./lite_data/val/",
        presets_torch.ClassificationPresetEval(
            crop_size=224, resize_size=256),
        is_valid_file=None)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=4,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True)
    return dataset_test, data_loader_test


def test_data_pipeline():
    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline()
    torch_dataset, torch_dataloader = build_torch_data_pipeline()

    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("length", np.array(len(paddle_dataset)))
    logger_torch_data.add("length", np.array(len(torch_dataset)))

    for idx, (paddle_batch, torch_batch
              ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx >= 5:
            break
        logger_paddle_data.add(f"dataloader_{idx}", paddle_batch[0].numpy())
        logger_torch_data.add(f"dataloader_{idx}",
                              torch_batch[0].detach().cpu().numpy())
    logger_paddle_data.save("./result/data_paddle.npy")
    logger_torch_data.save("./result/data_ref.npy")


if __name__ == "__main__":
    test_data_pipeline()

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/data_ref.npy")
    paddle_info = diff_helper.load_info("./result/data_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/data_diff.log")
