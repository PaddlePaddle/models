import numpy as np
import paddle
import torch

import mobilenetv3_paddle.paddlevision as paddlevision
import mobilenetv3_ref.torchvision as torchvision
import mobilenetv3_paddle.presets as presets_paddle
import mobilenetv3_ref.presets as presets_torch


def gen_fake_data():
    fake_data = np.random.rand(1, 3, 224, 224).astype(np.float32) - 0.5
    fake_label = np.arange(1).astype(np.int64)
    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)


def evaluate(image, labels, model, acc, tag, reprod_logger):
    model.eval()
    output = model(image)

    accracy = acc(output, labels, topk=(1, 5))

    reprod_logger.add("acc_top1", np.array(accracy[0]))
    reprod_logger.add("acc_top5", np.array(accracy[1]))

    reprod_logger.save("./result/metric_{}.npy".format(tag))


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
