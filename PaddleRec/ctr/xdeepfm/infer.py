import logging
import numpy as np
import pickle
import os
import paddle
import paddle.fluid as fluid

from args import parse_args
from criteo_reader import CriteoDataset
import network_conf
import utils

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fluid')
logger.setLevel(logging.INFO)


def infer():
    args = parse_args()
    print(args)

    if args.use_gpu == 1:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    inference_scope = fluid.Scope()

    test_files = [
        os.path.join(args.test_data_dir, x)
        for x in os.listdir(args.test_data_dir)
    ]
    criteo_dataset = CriteoDataset()
    test_reader = fluid.io.batch(
        criteo_dataset.test(test_files), batch_size=args.batch_size)

    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    cur_model_path = os.path.join(args.model_output_dir,
                                  'epoch_' + args.test_epoch, "checkpoint")

    with fluid.scope_guard(inference_scope):
        with fluid.framework.program_guard(test_program, startup_program):
            loss, auc, data_list, auc_states = eval(
                'network_conf.' + args.model_name)(
                    args.embedding_size, args.num_field, args.num_feat,
                    args.layer_sizes_dnn, args.act, args.reg,
                    args.layer_sizes_cin)

            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(feed_list=data_list, place=place)

            exe.run(startup_program)
            fluid.load(fluid.default_main_program(), cur_model_path)

            for var in auc_states:  # reset auc states
                set_zero(var.name, scope=inference_scope, place=place)

            loss_all = 0
            num_ins = 0
            for batch_id, data_test in enumerate(test_reader()):
                loss_val, auc_val = exe.run(test_program,
                                            feed=feeder.feed(data_test),
                                            fetch_list=[loss.name, auc.name])

                num_ins += len(data_test)
                loss_all += loss_val * len(data_test)
                logger.info('TEST --> batch: {} loss: {} auc_val: {}'.format(
                    batch_id, loss_all / num_ins, auc_val))

            print(
                'The last log info is the total Logloss and AUC for all test data. '
            )


def set_zero(var_name,
             scope=fluid.global_scope(),
             place=fluid.CPUPlace(),
             param_type="int64"):
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


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    utils.check_version()
    infer()
