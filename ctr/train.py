import argparse
import logging
import gzip
import paddle.v2 as paddle
from data_provider import field_index, detect_dataset, AvazuDataset
from model import CTRmodel

parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
parser.add_argument(
    '--train_data_path',
    type=str,
    required=True,
    help="path of training dataset")
parser.add_argument(
    '--batch_size',
    type=int,
    default=10000,
    help="size of mini-batch (default:10000)")
parser.add_argument(
    '--test_set_size',
    type=int,
    default=10000,
    help="size of the validation dataset(default: 10000)")
parser.add_argument(
    '--num_passes', type=int, default=10, help="number of passes to train")
parser.add_argument(
    '--num_lines_to_detact',
    type=int,
    default=500000,
    help="number of records to detect dataset's meta info")
parser.add_argument(
    '--model_output_prefix',
    type=str,
    default='./ctr_models',
    help='prefix of path for model to store (default: ./ctr_models)')

args = parser.parse_args()

dnn_layer_dims = [128, 64, 32, 1]
data_meta_info = detect_dataset(args.train_data_path, args.num_lines_to_detact)

logging.warning('detect categorical fields in dataset %s' %
                args.train_data_path)
for key, item in data_meta_info.items():
    logging.warning('    - {}\t{}'.format(key, item))

paddle.init(use_gpu=False, trainer_count=1)

model = CTRmodel(dnn_layer_dims, data_meta_info['dnn_input'],
                 data_meta_info['lr_input'])

# ==============================================================================
#                   cost and train period
# ==============================================================================


def train():

    params = paddle.parameters.create(model.train_cost)
    optimizer = paddle.optimizer.AdaGrad()

    trainer = paddle.trainer.SGD(
        cost=model.train_cost, parameters=params, update_equation=optimizer)

    dataset = AvazuDataset(
        args.train_data_path, n_records_as_test=args.test_set_size)

    def __event_handler__(event):
        if isinstance(event, paddle.event.EndIteration):
            num_samples = event.batch_id * args.batch_size
            if event.batch_id % 100 == 0:
                logging.warning("Pass %d, Samples %d, Cost %f" %
                                (event.pass_id, num_samples, event.cost))

            if event.batch_id % 1000 == 0:
                result = trainer.test(
                    reader=paddle.batch(
                        dataset.test, batch_size=args.batch_size),
                    feeding=field_index)
                logging.warning("Test %d-%d, Cost %f" %
                                (event.pass_id, event.batch_id, result.cost))

                path = "{}-pass-{}-batch-{}-test-{}.tar.gz".format(
                    args.model_output_prefix, event.pass_id, event.batch_id,
                    result.cost)
                with gzip.open(path, 'w') as f:
                    params.to_tar(f)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(dataset.train, buf_size=500),
            batch_size=args.batch_size),
        feeding=field_index,
        event_handler=__event_handler__,
        num_passes=args.num_passes)


if __name__ == '__main__':
    train()
