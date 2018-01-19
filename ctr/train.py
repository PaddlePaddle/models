import argparse
import gzip

import reader
import paddle.v2 as paddle
from utils import logger, ModelType
from network_conf import CTRmodel


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        required=True,
        help="path of training dataset")
    parser.add_argument(
        '--test_data_path', type=str, help='path of testing dataset')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10000,
        help="size of mini-batch (default:10000)")
    parser.add_argument(
        '--num_passes', type=int, default=10, help="number of passes to train")
    parser.add_argument(
        '--model_output_prefix',
        type=str,
        default='./ctr_models',
        help='prefix of path for model to store (default: ./ctr_models)')
    parser.add_argument(
        '--data_meta_file',
        type=str,
        required=True,
        help='path of data meta info file', )
    parser.add_argument(
        '--model_type',
        type=int,
        required=True,
        default=ModelType.CLASSIFICATION,
        help='model type, classification: %d, regression %d (default classification)'
        % (ModelType.CLASSIFICATION, ModelType.REGRESSION))

    return parser.parse_args()


dnn_layer_dims = [128, 64, 32, 1]

# ==============================================================================
#                   cost and train period
# ==============================================================================


def train():
    args = parse_args()
    args.model_type = ModelType(args.model_type)
    paddle.init(use_gpu=False, trainer_count=1)
    dnn_input_dim, lr_input_dim = reader.load_data_meta(args.data_meta_file)

    # create ctr model.
    model = CTRmodel(
        dnn_layer_dims,
        dnn_input_dim,
        lr_input_dim,
        model_type=args.model_type,
        is_infer=False)

    params = paddle.parameters.create(model.train_cost)
    optimizer = paddle.optimizer.AdaGrad()

    trainer = paddle.trainer.SGD(cost=model.train_cost,
                                 parameters=params,
                                 update_equation=optimizer)

    dataset = reader.Dataset()

    def __event_handler__(event):
        if isinstance(event, paddle.event.EndIteration):
            num_samples = event.batch_id * args.batch_size
            if event.batch_id % 100 == 0:
                logger.warning("Pass %d, Samples %d, Cost %f, %s" % (
                    event.pass_id, num_samples, event.cost, event.metrics))

            if event.batch_id % 1000 == 0:
                if args.test_data_path:
                    result = trainer.test(
                        reader=paddle.batch(
                            dataset.test(args.test_data_path),
                            batch_size=args.batch_size),
                        feeding=reader.feeding_index)
                    logger.warning("Test %d-%d, Cost %f, %s" %
                                   (event.pass_id, event.batch_id, result.cost,
                                    result.metrics))

                path = "{}-pass-{}-batch-{}-test-{}.tar.gz".format(
                    args.model_output_prefix, event.pass_id, event.batch_id,
                    result.cost)
                with gzip.open(path, 'w') as f:
                    trainer.save_parameter_to_tar(f)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                dataset.train(args.train_data_path), buf_size=500),
            batch_size=args.batch_size),
        feeding=reader.feeding_index,
        event_handler=__event_handler__,
        num_passes=args.num_passes)


if __name__ == '__main__':
    train()
