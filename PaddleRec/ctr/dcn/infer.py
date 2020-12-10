import logging
import random

import numpy as np
import pickle

# disable gpu training for this example
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import paddle
import paddle.fluid as fluid

from config import parse_args
from reader import CriteoDataset
from network import DCN
from collections import OrderedDict
import utils

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fluid')
logger.setLevel(logging.INFO)


def infer():
    args = parse_args()
    print(args)

    place = fluid.CPUPlace()
    inference_scope = fluid.Scope()

    test_valid_files = [
        os.path.join(args.test_valid_data_dir, fname)
        for fname in next(os.walk(args.test_valid_data_dir))[2]
    ]
    test_files = random.sample(test_valid_files,
                               int(len(test_valid_files) * 0.5))
    if not test_files:
        test_files = test_valid_files
    print('test files num {}'.format(len(test_files)))

    criteo_dataset = CriteoDataset()
    criteo_dataset.setup(args.vocab_dir)
    test_reader = criteo_dataset.test_reader(test_files, args.batch_size, 100)

    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    cur_model_path = os.path.join(args.model_output_dir,
                                  'epoch_' + args.test_epoch, "checkpoint")

    with fluid.scope_guard(inference_scope):
        with fluid.framework.program_guard(test_program, startup_program):
            cat_feat_dims_dict = OrderedDict()
            for line in open(args.cat_feat_num):
                spls = line.strip().split()
                assert len(spls) == 2
                cat_feat_dims_dict[spls[0]] = int(spls[1])
            dcn_model = DCN(args.cross_num, args.dnn_hidden_units,
                            args.l2_reg_cross, args.use_bn, args.clip_by_norm,
                            cat_feat_dims_dict, args.is_sparse)
            dcn_model.build_network(is_test=True)

            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(
                feed_list=dcn_model.data_list, place=place)

            exe.run(startup_program)
            fluid.load(fluid.default_main_program(), cur_model_path)

            for var in dcn_model.auc_states:  # reset auc states
                set_zero(var.name, scope=inference_scope, place=place)

            loss_all = 0
            num_ins = 0
            for batch_id, data_test in enumerate(test_reader()):
                loss_val, auc_val = exe.run(test_program,
                                            feed=feeder.feed(data_test),
                                            fetch_list=[
                                                dcn_model.avg_logloss.name,
                                                dcn_model.auc_var.name
                                            ])
                # num_ins += len(data_test)
                num_ins += 1
                loss_all += loss_val
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
