import logging
import argparse
import paddle.v2 as paddle
import gzip
from model import Model
from data_provider import get_file_list, AsciiDic, ImageDataset

parser = argparse.ArgumentParser(description="PaddlePaddle CTC example")
parser.add_argument(
    '--image_shape',
    type=str,
    required=True,
    help="image's shape, format is like '173,46'")
parser.add_argument(
    '--train_file_list',
    type=str,
    required=True,
    help='path of the file which contains path list of train image files')
parser.add_argument(
    '--test_file_list',
    type=str,
    required=True,
    help='path of the file which contains path list of test image files')
parser.add_argument(
    '--batch_size', type=int, default=5, help='size of a mini-batch')
parser.add_argument(
    '--model_output_prefix',
    type=str,
    default='model.ctc',
    help='prefix of path for model to store (default: ./model.ctc)')
parser.add_argument(
    '--trainer_count', type=int, default=4, help='number of training threads')
parser.add_argument(
    '--save_period_by_batch',
    type=int,
    default=50,
    help='save model to disk every N batches')
parser.add_argument(
    '--num_passes',
    type=int,
    default=1,
    help='number of passes to train (default: 1)')

args = parser.parse_args()

image_shape = tuple(map(int, args.image_shape.split(',')))

print 'image_shape', image_shape
print 'batch_size', args.batch_size
print 'train_file_list', args.train_file_list
print 'test_file_list', args.test_file_list

train_generator = get_file_list(args.train_file_list)
test_generator = get_file_list(args.test_file_list)
infer_generator = None

dataset = ImageDataset(
    train_generator,
    test_generator,
    infer_generator,
    fixed_shape=image_shape,
    is_infer=False)

paddle.init(use_gpu=True, trainer_count=args.trainer_count)

model = Model(AsciiDic().size(), image_shape, is_infer=False)
params = paddle.parameters.create(model.cost)
optimizer = paddle.optimizer.Momentum(momentum=0)
trainer = paddle.trainer.SGD(
    cost=model.cost, parameters=params, update_equation=optimizer)


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "Pass %d, batch %d, Samples %d, Cost %f" % (
                event.pass_id, event.batch_id, event.batch_id * args.batch_size,
                event.cost)

        if event.batch_id > 0 and event.batch_id % args.save_period_by_batch == 0:
            result = trainer.test(
                reader=paddle.batch(dataset.test, batch_size=10),
                feeding={'image': 0,
                         'label': 1})
            print "Test %d-%d, Cost %f " % (event.pass_id, event.batch_id,
                                            result.cost)

            path = "{}-pass-{}-batch-{}-test-{}.tar.gz".format(
                args.model_output_prefix, event.pass_id, event.batch_id,
                result.cost)
            with gzip.open(path, 'w') as f:
                params.to_tar(f)


trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(dataset.train, buf_size=500),
        batch_size=args.batch_size),
    feeding={'image': 0,
             'label': 1},
    event_handler=event_handler,
    num_passes=args.num_passes)
