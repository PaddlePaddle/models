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
    test_reader = paddle.batch(
        criteo_dataset.test(test_files), batch_size=args.batch_size)

    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    cur_model_path = os.path.join(args.model_output_dir,
                                  'epoch_' + args.test_epoch)

    with fluid.scope_guard(inference_scope):
        with fluid.framework.program_guard(test_program, startup_program):
            loss, auc, data_list = eval('network_conf.' + args.model_name)(
                args.embedding_size, args.num_field, args.num_feat,
                args.layer_sizes_dnn, args.act, args.reg, args.layer_sizes_cin)

            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(feed_list=data_list, place=place)
            fluid.io.load_persistables(
                executor=exe,
                dirname=cur_model_path,
                main_program=fluid.default_main_program())

            auc_states_names = ['_generated_var_2', '_generated_var_3']
            for name in auc_states_names:
                param = inference_scope.var(name).get_tensor()
                param_array = np.zeros(param._get_dims()).astype("int64")
                param.set(param_array, place)

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


if __name__ == '__main__':
    utils.check_version()
    infer()
