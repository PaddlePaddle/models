import numpy as np
import os
import paddle.fluid as fluid
from net import wide_deep
import logging
import paddle
import args
import utils
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def train(args, train_data_path):
    wide_deep_model = wide_deep()
    inputs = wide_deep_model.input_data()
    train_data_generator = utils.Dataset()
    train_reader = fluid.io.batch(train_data_generator.train(train_data_path), batch_size=args.batch_size)
    
    loss, acc, auc, batch_auc, auc_states  = wide_deep_model.model(inputs, args.hidden1_units, args.hidden2_units, args.hidden3_units)
    optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.01)
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
            loss_val, acc_val, auc_val = exe.run(program=fluid.default_main_program(),
                    feed=data,
                    fetch_list=[loss.name, acc.name, auc.name],
                    return_numpy=True)
            end = time.time()
            logger.info("epoch:{}, batch_time:{:.5f}s, loss:{:.5f}, acc:{:.5f}, auc:{:.5f}".format(epoch, end-begin, np.array(loss_val)[0], 
                    np.array(acc_val)[0], np.array(auc_val)[0]))

        model_dir = os.path.join(args.model_dir, 'epoch_' + str(epoch + 1), "checkpoint")
        main_program = fluid.default_main_program()
        fluid.io.save(main_program,model_dir)
  
if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = args.parse_args()
    train(args, args.train_data_path)
