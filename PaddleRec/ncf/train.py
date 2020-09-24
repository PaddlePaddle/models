import numpy as np
import os
import paddle.fluid as fluid
from gmf import GMF
from mlp import MLP
from neumf import NeuMF
from Dataset import Dataset
import logging
import paddle
import args
import utils
import time
from evaluate import evaluate_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def train(args, train_data_path):
    print("use_gpu:{}, NeuMF:{}, epochs:{}, batch_size:{}, num_factors:{}, num_neg:{}, lr:{}, model_dir:{}, layers:{}".format(
        args.use_gpu, args.NeuMF, args.epochs, args.batch_size, args.num_factors, args.num_neg, args.lr, args.model_dir, args.layers))
    dataset = Dataset(args.path + args.dataset)
    testRatings, testNegatives = dataset.testRatings, dataset.testNegatives

    train_data_generator = utils.Dataset()
    train_reader = fluid.io.batch(train_data_generator.train(train_data_path, True), batch_size=args.batch_size)
    
    inputs = utils.input_data(True)
    if args.GMF:
        model = GMF()
        loss, pred = model.net(inputs, args.num_users, args.num_items, args.num_factors)
    elif args.MLP:
        model = MLP()
        loss, pred = model.net(inputs, args.num_users, args.num_items, args.layers)
    elif args.NeuMF:
        model = NeuMF()
        loss, pred = model.net(inputs, args.num_users, args.num_items, args.num_factors, args.layers)

    optimizer = fluid.optimizer.AdamOptimizer(args.lr)
    optimizer.minimize(loss)
    
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    loader = fluid.io.DataLoader.from_generator(
        feed_list=inputs, capacity=args.batch_size, iterable=True)
    loader.set_sample_list_generator(train_reader, places=place)
    
    for epoch in range(args.epochs):

        for batch_id, data in enumerate(loader()):
            begin = time.time()
            loss_val = exe.run(program=fluid.default_main_program(),
                    feed=data,
                    fetch_list=[loss.name],
                    return_numpy=True)
            end = time.time()
            logger.info("epoch: {}, batch_id: {}, batch_time: {:.5f}s, loss: {:.5f}".format(epoch, batch_id, end - begin, np.array(loss_val)[0][0]))

        save_dir = "%s/epoch_%d" % (args.model_dir, epoch)
        feed_var_names = ["user_input", "item_input"]
        fetch_vars = [pred]
        fluid.io.save_inference_model(save_dir, feed_var_names, fetch_vars, exe)
 
if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = args.parse_args()
    train(args, args.train_data_path)
