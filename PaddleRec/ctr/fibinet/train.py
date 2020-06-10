import paddle.fluid as fluid
import logging
import args
import time
import os
import numpy as np
import random
from net import Fibinet
import feed_generator as generator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def get_dataset(inputs, args):
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python dataset_generator.py")
    dataset.set_batch_size(args.batch_size)
    thread_num = int(args.cpu_num)
    dataset.set_thread(thread_num)
    file_list = [
        os.path.join(args.train_files_path, x) for x in os.listdir(args.train_files_path)
    ]
    logger.info("file list: {}".format(file_list))

    return dataset, file_list

def train(args):
    fibinet_model = Fibinet()
    inputs = fibinet_model.input_data(args.dense_feature_dim)
    data_generator = generator.CriteoDataset(args.sparse_feature_dim)

    file_list = [os.path.join(args.train_files_path, x) for x in os.listdir(args.train_files_path)]
    train_reader = fluid.io.batch(data_generator.train(file_list), batch_size=args.batch_size)

    avg_cost, auc_val, batch_auc, auc_states = fibinet_model.net(inputs, args.sparse_feature_dim, args.embedding_size, 
                                            args.reduction_ratio, args.bilinear_type, args.dropout_rate)

    optimizer = fluid.optimizer.Adam(args.learning_rate)
    optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    loader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=args.batch_size, iterable=True)
    loader.set_sample_list_generator(train_reader, places=place)

    for epoch in range(args.epochs):
        for batch_id, data in enumerate(loader()):
            begin = time.time()
            loss_data, auc = exe.run(program=fluid.default_main_program(),
                                    feed=data,
                                    fetch_list=[avg_cost.name, auc_val.name],
                                    return_numpy=True)
            end = time.time()
            logger.info("epoch_id: {}, batch_id: {}, batch_time: {:.5f}s, loss: {:.5f}, auc: {:.5f}".format(
                epoch, batch_id, end-begin, float(np.array(loss_data)), np.array(auc)[0]))

        model_dir = os.path.join(args.model_dir, 'epoch_' + str(epoch + 1), "checkpoint")
        main_program = fluid.default_main_program()
        fluid.io.save(main_program, model_dir)
            
if __name__ == '__main__':
    args = args.parse_args()
    logger.info("use_gpu:{}, train_files_path: {}, model_dir: {}, learning_rate: {}, batch_size: {}, epochs: {}, reduction_ratio: {}, dropout_rate: {}, embedding_size: {}".format(
        args.use_gpu, args.train_files_path, args.model_dir, args.learning_rate, args.batch_size, args.epochs, args.reduction_ratio, args.dropout_rate, args.embedding_size))

    train(args)