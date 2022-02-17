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


def run_infer(args,test_data_path):
    wide_deep_model = wide_deep()
    test_data_generator = utils.Dataset()
    test_reader = fluid.io.batch(test_data_generator.test(test_data_path), batch_size=args.batch_size)
    inference_scope = fluid.Scope()
    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    
    cur_model_path = os.path.join(args.model_dir, 'epoch_' + str(args.test_epoch), "checkpoint")

    with fluid.scope_guard(inference_scope):
        with fluid.framework.program_guard(test_program, startup_program):
            inputs = wide_deep_model.input_data()
            place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
            loss, acc, auc, batch_auc, auc_states = wide_deep_model.model(inputs, args.hidden1_units, args.hidden2_units, args.hidden3_units)
            exe = fluid.Executor(place)

            fluid.load(fluid.default_main_program(), cur_model_path,exe)
            loader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=args.batch_size, iterable=True)
            loader.set_sample_list_generator(test_reader, places=place)
            
            for var in auc_states:  # reset auc states
                set_zero(var.name, scope=inference_scope, place=place)

            mean_acc = []
            mean_auc = []
            for batch_id, data in enumerate(loader()):
                begin = time.time()
                acc_val,auc_val = exe.run(program=test_program,
                                        feed=data,
                                        fetch_list=[acc.name, auc.name],
                                        return_numpy=True
                                        )
                mean_acc.append(np.array(acc_val)[0])
                mean_auc.append(np.array(auc_val)[0])
                end = time.time()
                logger.info("batch_id: {}, batch_time: {:.5f}s, acc: {:.5f}, auc: {:.5f}".format(
                            batch_id, end-begin, np.array(acc_val)[0], np.array(auc_val)[0]))
                            
            logger.info("mean_acc:{:.5f}, mean_auc:{:.5f}".format(np.mean(mean_acc), np.mean(mean_auc)))
                
if __name__ == "__main__":
    import paddle
    paddle.enable_static()
  
    args = args.parse_args()
    run_infer(args, args.test_data_path)
              