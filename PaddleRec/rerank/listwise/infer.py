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
    def get_test_data(self):
        return self._reader_creator()

def set_zero(var_name, scope=fluid.global_scope(), place=fluid.CPUPlace(), param_type="int64"):
    """
    Set tensor of a Variable to zero.
    Args:
        var_name(str): name of Variable
        scope(Scope): Scope object, default is fluid.global_scope()
        place(Place): Place object, default is fluid.CPUPlace()
        param_type(str): param data type, default is int64
    """
    param = scope.var(var_name).get_tensor()
    param_array = np.zeros(param._get_dims()).astype(param_type)
    param.set(param_array, place)

def run_infer(args):
    model = BiRNN()
    inference_scope = fluid.Scope()
    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    cur_model_path = os.path.join(args.model_dir, 'epoch_' + str(args.test_epoch), "checkpoint")
    with fluid.scope_guard(inference_scope):
        with fluid.framework.program_guard(test_program, startup_program):
            place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
            inputs = model.input_data(args.item_len)
            loss, auc_val, batch_auc, auc_states = model.net(inputs, args.hidden_size, args.batch_size*args.sample_size, args.item_vocab, args.embd_dim)
            exe = fluid.Executor(place)

            fluid.load(fluid.default_main_program(), cur_model_path, exe)
            for var in auc_states:  # reset auc states
                set_zero(var.name, scope=inference_scope, place=place)

            test_data_generator = Dataset()
            test_reader = fluid.io.batch(test_data_generator.get_test_data(), batch_size=args.batch_size)
            loader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=args.batch_size, iterable=True)
            loader.set_sample_list_generator(test_reader, places=place)

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

if __name__ == "__main__":
    args = args.parse_args()
    logger.info("use_gpu: {}, model_dir: {}, test_epoch: {}".format(args.use_gpu, args.model_dir, args.test_epoch))
    run_infer(args)