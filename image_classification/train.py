import gzip
import argparse

import paddle.v2.dataset.flowers as flowers
import paddle.v2 as paddle
import reader
import vgg
import resnet
import alexnet
import googlenet
import inception_resnet_v2

DATA_DIM = 3 * 224 * 224  # Use 3 * 331 * 331 or 3 * 299 * 299 for Inception-ResNet-v2.
CLASS_DIM = 102
BATCH_SIZE = 128


def main():
    # parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model',
        help='The model for image classification',
        choices=[
            'alexnet', 'vgg13', 'vgg16', 'vgg19', 'resnet', 'googlenet',
            'inception-resnet-v2'
        ])
    args = parser.parse_args()

    # PaddlePaddle init
    paddle.init(use_gpu=True, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))
    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(CLASS_DIM))

    extra_layers = None
    learning_rate = 0.01
    if args.model == 'alexnet':
        out = alexnet.alexnet(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg13':
        out = vgg.vgg13(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg16':
        out = vgg.vgg16(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg19':
        out = vgg.vgg19(image, class_dim=CLASS_DIM)
    elif args.model == 'resnet':
        out = resnet.resnet_imagenet(image, class_dim=CLASS_DIM)
        learning_rate = 0.1
    elif args.model == 'googlenet':
        out, out1, out2 = googlenet.googlenet(image, class_dim=CLASS_DIM)
        loss1 = paddle.layer.cross_entropy_cost(
            input=out1, label=lbl, coeff=0.3)
        paddle.evaluator.classification_error(input=out1, label=lbl)
        loss2 = paddle.layer.cross_entropy_cost(
            input=out2, label=lbl, coeff=0.3)
        paddle.evaluator.classification_error(input=out2, label=lbl)
        extra_layers = [loss1, loss2]
    elif args.model == 'inception-resnet-v2':
        assert DATA_DIM == 3 * 331 * 331 or DATA_DIM == 3 * 299 * 299
        out = inception_resnet_v2.inception_resnet_v2(
            image, class_dim=CLASS_DIM, dropout_rate=0.5, data_dim=DATA_DIM)

    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # Create parameters
    parameters = paddle.parameters.create(cost)

    # Create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 *
                                                         BATCH_SIZE),
        learning_rate=learning_rate / BATCH_SIZE,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=128000 * 35,
        learning_rate_schedule="discexp", )

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            flowers.train(),
            # To use other data, replace the above line with:
            # reader.train_reader('train.list'),
            buf_size=1000),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        flowers.valid(),
        # To use other data, replace the above line with:
        # reader.test_reader('val.list'),
        batch_size=BATCH_SIZE)

    # Create trainer
    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=optimizer,
        extra_layers=extra_layers)

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        if isinstance(event, paddle.event.EndPass):
            with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)

            result = trainer.test(reader=test_reader)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    trainer.train(
        reader=train_reader, num_passes=200, event_handler=event_handler)


if __name__ == '__main__':
    main()
