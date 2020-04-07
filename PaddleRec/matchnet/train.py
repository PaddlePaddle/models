from util import logger, check_version
import paddle.fluid as fluid
import reader
import time
import paddle
import argparse
from model import Model


def parse_args():
    parser = argparse.ArgumentParser("matchnet")
    parser.add_argument("--train_file", type=str, help="Training file")
    parser.add_argument("--valid_file", type=str, help="Validation file")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default='model_output',
        help="Model output folder")
    parser.add_argument(
        "--user_slots", type=int, default=1, help="Number of query slots")
    parser.add_argument(
        "--title_slots", type=int, default=1, help="Number of title slots")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Default Dimension of Embedding")
    parser.add_argument(
        "--sparse_feature_dim",
        type=int,
        default=1000001,
        help="Sparse feature hashing space"
        "for index processing")
    parser.add_argument(
        "--random_ratio",
        type=int,
        default=1,
        help="random ratio for negative samples.")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='If set, run the task with continuous evaluation logs.')
    return parser.parse_args()


def train(args):
    if args.enable_ce:
        SEED = 102
        fluid.default_startup_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED

    dataset = reader.SyntheticDataset(args.sparse_feature_dim, args.user_slots,
                                      args.title_slots)
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            dataset.train(), buf_size=args.batch_size * 100),
        batch_size=args.batch_size)
    place = fluid.CPUPlace()
    model = Model(args.user_slots, args.title_slots, args.title_slots,
                  args.sparse_feature_dim, args.embedding_dim,
                  args.random_ratio)
    with fluid.program_guard(model._train_program, model._startup_program):
        with fluid.unique_name.guard():
            optimizer = fluid.optimizer.Adam(learning_rate=float(args.lr))
            optimizer.minimize(model.avg_cost)

    exe = fluid.Executor(place)
    exe.run(model._startup_program)
    loader = fluid.io.DataLoader.from_generator(
        feed_list=model._all_slots, capacity=10000, iterable=True)
    loader.set_sample_list_generator(train_reader, places=place)

    total_time = 0
    ce_info = []
    for pass_id in range(args.epochs):
        start_time = time.time()
        for batch_id, data in enumerate(loader()):
            loss_val, correct_val, wrong_val, pn_val = exe.run(
                model._train_program,
                feed=data,
                fetch_list=[
                    model.avg_cost, model.correct, model.wrong, model.pn
                ])
            logger.info(
                "TRAIN --> pass: {} batch_id: {} avg_cost: {}, correct: {}, wrong: {}, pn: {}"
                .format(pass_id, batch_id, loss_val, correct_val, wrong_val,
                        pn_val))
            ce_info.append(loss_val[0])
        end_time = time.time()
        total_time += end_time - start_time
        fluid.io.save_inference_model(
            args.model_output_dir, [val.name for val in model._all_slots],
            [model.avg_cost, model.correct, model.wrong, model.pn], exe)


if __name__ == "__main__":
    check_version()
    args = parse_args()
    train(args)
