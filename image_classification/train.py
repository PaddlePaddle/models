import gzip
import paddle.v2 as paddle
import reader
import vgg
import resnet
import alexnet
import googlenet
import argparse
import os

DATA_DIM = 3 * 224 * 224
CLASS_DIM = 100
BATCH_SIZE = 128


def main():
    # parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_dir',
        help='The data directory which contains train.list and val.list')
    parser.add_argument(
        'model',
        help='The model for image classification',
        choices=['alexnet', 'vgg13', 'vgg16', 'vgg19', 'resnet', 'googlenet'])
    args = parser.parse_args()

    # PaddlePaddle init
    paddle.init(use_gpu=True, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))
    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(CLASS_DIM))

    extra_layers = None
    if args.model == 'alexnet':
        net = alexnet.alexnet(image)
    elif args.model == 'vgg13':
        net = vgg.vgg13(image)
    elif args.model == 'vgg16':
        net = vgg.vgg16(image)
    elif args.model == 'vgg19':
        net = vgg.vgg19(image)
    elif args.model == 'resnet':
        net = resnet.resnet_imagenet(image)
    elif args.model == 'googlenet':
        net, fc_o1, fc_o2 = googlenet.googlenet(image)
        out1 = paddle.layer.fc(
            input=fc_o1, size=CLASS_DIM, act=paddle.activation.Softmax())
        loss1 = paddle.layer.cross_entropy_cost(
            input=out1, label=lbl, coeff=0.3)
        paddle.evaluator.classification_error(input=out1, label=lbl)
        out2 = paddle.layer.fc(
            input=fc_o2, size=CLASS_DIM, act=paddle.activation.Softmax())
        loss2 = paddle.layer.cross_entropy_cost(
            input=out2, label=lbl, coeff=0.3)
        paddle.evaluator.classification_error(input=out2, label=lbl)
        extra_layers = [loss1, loss2]

    out = paddle.layer.fc(
        input=net, size=CLASS_DIM, act=paddle.activation.Softmax())
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # Create parameters
    parameters = paddle.parameters.create(cost)

    # Create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 *
                                                         BATCH_SIZE),
        learning_rate=0.001 / BATCH_SIZE,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=128000 * 35,
        learning_rate_schedule="discexp", )

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.test_reader(os.path.join(args.data_dir, 'train.list')),
            buf_size=1000),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        reader.train_reader(os.path.join(args.data_dir, 'val.list')),
        batch_size=BATCH_SIZE)

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        if isinstance(event, paddle.event.EndPass):
            with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)

            result = trainer.test(reader=test_reader)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    # Create trainer
    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=optimizer,
        extra_layers=extra_layers)

    trainer.train(
        reader=train_reader, num_passes=200, event_handler=event_handler)


if __name__ == '__main__':
    main()
