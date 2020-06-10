import numpy as np
import os
import paddle.fluid as fluid
from net import Fibinet
import feed_generator as generator
import logging
import args
import utils
import time
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def set_zero(var_name,scope=fluid.global_scope(), place=fluid.CPUPlace(),param_type="int64"):
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
    fibinet_model = Fibinet()
    data_generator = generator.CriteoDataset(args.sparse_feature_dim)

    file_list = [os.path.join(args.test_files_path, x) for x in os.listdir(args.test_files_path)]
    test_reader = fluid.io.batch(data_generator.test(file_list), batch_size=args.batch_size)

    inference_scope = fluid.Scope()
    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()

    cur_model_path = os.path.join(args.model_dir, 'epoch_' + str(args.test_epoch), "checkpoint")

    with fluid.scope_guard(inference_scope):
        with fluid.framework.program_guard(test_program, startup_program):
            inputs = fibinet_model.input_data(args.dense_feature_dim)
            place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
            avg_cost, auc_val, batch_auc, auc_states = fibinet_model.net(inputs, args.sparse_feature_dim, args.embedding_size, args.reduction_ratio, 
                                                        args.bilinear_type, args.dropout_rate)
            exe = fluid.Executor(place)
            fluid.load(fluid.default_main_program(), cur_model_path, exe)
            loader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=args.batch_size, iterable=True)
            loader.set_sample_list_generator(test_reader, places=place)

            for var in auc_states:  # reset auc states
                set_zero(var.name, scope=inference_scope, place=place)

            for batch_id, data in enumerate(loader()):
                begin = time.time()
                auc = exe.run(program=test_program,
                                        feed=data,
                                        fetch_list=[auc_val.name],
                                        return_numpy=True)
                end = time.time()
                logger.info("batch_id: {}, batch_time: {:.5f}s, auc: {:.5f}".format(
                    batch_id, end-begin, np.array(auc)[0]))
        

if __name__ == "__main__":
    args = args.parse_args()
    logger.info("use_gpu:{}, test_files_path: {}, model_dir: {}, test_epoch: {}".format(
        args.use_gpu, args.test_files_path, args.model_dir, args.test_epoch))
    run_infer(args)
