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
import paddle
import paddle.fluid as fluid
import utils.utility as utility

def _calc_label_smoothing_loss(softmax_out, label, class_dim, epsilon):
    """Calculate label smoothing loss

    Returns:
        label smoothing loss
        
    """

    label_one_hot = fluid.layers.one_hot(input=label, depth=class_dim)
    smooth_label = fluid.layers.label_smooth(
        label=label_one_hot, epsilon=epsilon, dtype="float32")
    loss = fluid.layers.cross_entropy(
        input=softmax_out, label=smooth_label, soft_label=True)
    return loss


def _basic_model(data, model, args, is_train):
    image = data[0]
    label = data[1]
    if args.model == "ResNet50":
        image_data = (fluid.layers.cast(image, 'float16')
            if args.use_pure_fp16 and not args.use_dali else image)
        image_transpose = fluid.layers.transpose(
            image_data, [0, 2, 3, 1]) if args.data_format == 'NHWC' else image_data
        image_transpose.stop_gradient = image.stop_gradient
        net_out = model.net(input=image_transpose,
                            class_dim=args.class_dim,
                            data_format=args.data_format)
    else:
        net_out = model.net(input=image, class_dim=args.class_dim)
    if args.use_pure_fp16:
        net_out_fp32 = fluid.layers.cast(x=net_out, dtype="float32")
        softmax_out = fluid.layers.softmax(net_out_fp32, use_cudnn=False)
    else:
        softmax_out = fluid.layers.softmax(net_out, use_cudnn=False)

    if is_train and args.use_label_smoothing:
        cost = _calc_label_smoothing_loss(softmax_out, label, args.class_dim,
                                          args.label_smoothing_epsilon)

    else:
        cost = fluid.layers.cross_entropy(input=softmax_out, label=label)

    target_cost = (fluid.layers.reduce_sum(cost) if args.use_pure_fp16
        else fluid.layers.mean(cost))
    acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(
        input=softmax_out, label=label, k=min(5, args.class_dim))
    return [target_cost, acc_top1, acc_top5]


def _googlenet_model(data, model, args, is_train):
    """GoogLeNet model output, include avg_cost, acc_top1 and acc_top5
        
    Returns:
         GoogLeNet model output

    """
    image = data[0]
    label = data[1]

    out0, out1, out2 = model.net(input=image, class_dim=args.class_dim)
    cost0 = fluid.layers.cross_entropy(input=out0, label=label)
    cost1 = fluid.layers.cross_entropy(input=out1, label=label)
    cost2 = fluid.layers.cross_entropy(input=out2, label=label)

    avg_cost0 = fluid.layers.mean(x=cost0)
    avg_cost1 = fluid.layers.mean(x=cost1)
    avg_cost2 = fluid.layers.mean(x=cost2)

    avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
    acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(
        input=out0, label=label, k=min(5, args.class_dim))

    return [avg_cost, acc_top1, acc_top5]


def _mixup_model(data, model, args, is_train):
    """output of Mixup processing network, include avg_cost
    """
    image = data[0]
    y_a = data[1]
    y_b = data[2]
    lam = data[3]

    if args.model == "ResNet50":
        image_data = (fluid.layers.cast(image, 'float16')
            if args.use_pure_fp16 and not args.use_dali else image)
        image_transpose = fluid.layers.transpose(
            image_data, [0, 2, 3, 1]) if args.data_format == 'NHWC' else image_data
        image_transpose.stop_gradient = image.stop_gradient
        net_out = model.net(input=image_transpose,
                            class_dim=args.class_dim,
                            data_format=args.data_format)
    else:
        net_out = model.net(input=image, class_dim=args.class_dim)
    if args.use_pure_fp16:
        net_out_fp32 = fluid.layers.cast(x=net_out, dtype="float32")
        softmax_out = fluid.layers.softmax(net_out_fp32, use_cudnn=False)
    else:
        softmax_out = fluid.layers.softmax(net_out, use_cudnn=False)
    if not args.use_label_smoothing:
        loss_a = fluid.layers.cross_entropy(input=softmax_out, label=y_a)
        loss_b = fluid.layers.cross_entropy(input=softmax_out, label=y_b)
    else:
        loss_a = _calc_label_smoothing_loss(softmax_out, y_a, args.class_dim,
                                            args.label_smoothing_epsilon)
        loss_b = _calc_label_smoothing_loss(softmax_out, y_b, args.class_dim,
                                            args.label_smoothing_epsilon)

    if args.use_pure_fp16:
        target_loss_a = fluid.layers.reduce_sum(x=loss_a)
        target_loss_b = fluid.layers.reduce_sum(x=loss_b)
        cost = lam * target_loss_a + (1 - lam) * target_loss_b
        target_cost = fluid.layers.reduce_sum(x=cost)
    else:
        target_loss_a = fluid.layers.mean(x=loss_a)
        target_loss_b = fluid.layers.mean(x=loss_b)
        cost = lam * target_loss_a + (1 - lam) * target_loss_b
        target_cost = fluid.layers.mean(x=cost)
    return [target_cost]


def create_model(model, args, is_train):
    """Create model, include basic model, googlenet model and mixup model
    """
    data_loader, data = utility.create_data_loader(is_train, args)

    if args.model == "GoogLeNet":
        loss_out = _googlenet_model(data, model, args, is_train)
    else:
        if args.use_mixup and is_train:
            loss_out = _mixup_model(data, model, args, is_train)
        else:
            loss_out = _basic_model(data, model, args, is_train)
    return data_loader, loss_out
