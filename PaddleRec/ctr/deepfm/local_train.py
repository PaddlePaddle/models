from args import parse_args
import os
import paddle.fluid as fluid
import sys
from network_conf import ctr_deepfm_model
import time
import numpy
import pickle
import utils


def train():
    args = parse_args()
    # add ce
    if args.enable_ce:
        SEED = 102
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED

    print('---------- Configuration Arguments ----------')
    for key, value in args.__dict__.items():
        print(key + ':' + str(value))

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    loss, auc, data_list, auc_states = ctr_deepfm_model(
        args.embedding_size, args.num_field, args.num_feat, args.layer_sizes,
        args.act, args.reg)
    optimizer = fluid.optimizer.SGD(
        learning_rate=args.lr,
        regularization=fluid.regularizer.L2DecayRegularizer(args.reg))
    optimizer.minimize(loss)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(data_list)
    pipe_command = 'python criteo_reader.py {}'.format(args.feat_dict)
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(args.batch_size)
    dataset.set_thread(args.num_thread)
    train_filelist = [
        os.path.join(args.train_data_dir, x)
        for x in os.listdir(args.train_data_dir)
    ]

    print('---------------------------------------------')
    for epoch_id in range(args.num_epoch):
        start = time.time()
        dataset.set_filelist(train_filelist)
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=dataset,
            fetch_list=[loss, auc],
            fetch_info=['epoch %d batch loss' % (epoch_id + 1), "auc"],
            print_period=1000,
            debug=False)
        model_dir = os.path.join(args.model_output_dir,
                                 'epoch_' + str(epoch_id + 1))
        sys.stderr.write('epoch%d is finished and takes %f s\n' % (
            (epoch_id + 1), time.time() - start))
        main_program = fluid.default_main_program()
        fluid.io.save(main_program, model_dir)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    utils.check_version()
    train()
