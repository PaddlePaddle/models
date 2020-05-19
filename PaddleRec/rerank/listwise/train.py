import numpy as np
import os
import paddle.fluid as fluid
import logging
import args
import random
import time
from evaluator import BiRNN

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)
user_id = 0
class Dataset(object):
    def _reader_creator(self):
        def reader():
            global user_id
            user_slot_name = []
            for j in range(args.batch_size):
                user_slot_name.append([user_id])
                user_id += 1
            
            item_slot_name = np.random.randint(args.item_vocab, size=(args.batch_size, args.item_len)).tolist()
            lenght = [args.item_len]*args.batch_size
            label = np.random.randint(2, size=(args.batch_size, args.item_len)).tolist()
            output = []
            output.append(user_slot_name)
            output.append(item_slot_name)
            output.append(lenght)
            output.append(label)

            yield output
        return reader
    def get_train_data(self):
        return self._reader_creator()

def train(args):

    model = BiRNN()
    inputs = model.input_data(args.item_len)
    loss, auc_val, batch_auc, auc_states = model.net(inputs, args.hidden_size, args.batch_size*args.sample_size, args.item_vocab, args.embd_dim)

    optimizer = fluid.optimizer.Adam(learning_rate=args.base_lr, epsilon=1e-4)
    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    
    train_data_generator = Dataset()
    train_reader = fluid.io.batch(train_data_generator.get_train_data(), batch_size=args.batch_size)
    loader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=args.batch_size, iterable=True)
    loader.set_sample_list_generator(train_reader, places=place)

    for i in range(args.sample_size):
        for batch_id, data in enumerate(loader()):
            begin = time.time()
            loss_val, auc = exe.run(program=fluid.default_main_program(),
                    feed=data,
                    fetch_list=[loss.name, auc_val],
                    return_numpy=True)
            end = time.time()
            logger.info("batch_id: {}, batch_time: {:.5f}s, loss: {:.5f}, auc: {:.5f}".format(
                batch_id, end-begin, float(np.array(loss_val)), float(np.array(auc))))
        
    #save model
    model_dir = os.path.join(args.model_dir, 'epoch_' + str(1), "checkpoint")
    main_program = fluid.default_main_program()
    fluid.save(main_program, model_dir)

if __name__ == "__main__":
    args = args.parse_args()
    logger.info("use_gpu: {}, batch_size: {}, model_dir: {}, embd_dim: {}, hidden_size: {}, item_vocab: {}, user_vocab: {},\
    item_len: {}, sample_size: {}, base_lr: {}".format(args.use_gpu, args.batch_size, args.model_dir, args.embd_dim, 
     args.hidden_size, args.item_vocab, args.user_vocab, args.item_len, args.sample_size, args.base_lr))

    train(args)