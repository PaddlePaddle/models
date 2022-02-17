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

def train(args):

    model = BiRNN()
    inputs = model.input_data(args.item_len)
    loss, auc_val, batch_auc, auc_states = model.net(inputs, args.hidden_size, args.batch_size*args.sample_size, args.item_vocab, args.embd_dim)

    optimizer = fluid.optimizer.Adam(learning_rate=args.base_lr, epsilon=1e-4)
    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # Build a random data set.
    user_slot_names = []
    item_slot_names = []
    lens = []
    labels = []
    user_id = 0
    for i in range(args.sample_size):
        user_slot_name = []
        for j in range(args.batch_size):
            user_slot_name.append(user_id)
            user_id += 1
        user_slot_names.append(user_slot_name)

        item_slot_name = np.random.randint(args.item_vocab, size=(args.batch_size, args.item_len))
        item_slot_names.append(item_slot_name)
        lenght = np.array([args.item_len]*args.batch_size)
        lens.append(lenght)
        label = np.random.randint(2, size=(args.batch_size, args.item_len))
        labels.append(label)

    for epoch in range(args.epochs):
        for i in range(args.sample_size):
            begin = time.time()
            loss_val, auc = exe.run(fluid.default_main_program(),
                                feed={
                                    "user_slot_names": np.array(user_slot_names[i]).reshape(args.batch_size, 1),
                                    "item_slot_names": item_slot_names[i].astype('int64'),
                                    "lens": lens[i].astype('int64'),
                                    "labels": labels[i].astype('int64')
                                },
                                return_numpy=True,
                                fetch_list=[loss.name, auc_val])
            end = time.time()
            logger.info("epoch_id: {}, batch_time: {:.5f}s, loss: {:.5f}, auc: {:.5f}".format(
                epoch, end-begin, float(np.array(loss_val)), float(np.array(auc))))
        
        #save model
        model_dir = os.path.join(args.model_dir, 'epoch_' + str(epoch + 1), "checkpoint")
        main_program = fluid.default_main_program()
        fluid.save(main_program, model_dir)

if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = args.parse_args()
    train(args)