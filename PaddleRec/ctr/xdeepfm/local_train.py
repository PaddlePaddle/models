from args import parse_args
import os
import paddle.fluid as fluid
import sys
import network_conf
import time
import utils


def train():
    args = parse_args()
    print(args)
    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    loss, auc, data_list = eval('network_conf.' + args.model_name)(
        args.embedding_size, args.num_field, args.num_feat,
        args.layer_sizes_dnn, args.act, args.reg, args.layer_sizes_cin)
    optimizer = fluid.optimizer.SGD(
        learning_rate=args.lr,
        regularization=fluid.regularizer.L2DecayRegularizer(args.reg))
    optimizer.minimize(loss)

    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(data_list)
    dataset.set_pipe_command('python criteo_reader.py')
    dataset.set_batch_size(args.batch_size)
    dataset.set_filelist([
        os.path.join(args.train_data_dir, x)
        for x in os.listdir(args.train_data_dir)
    ])

    if args.use_gpu == 1:
        exe = fluid.Executor(fluid.CUDAPlace(0))
        dataset.set_thread(1)
    else:
        exe = fluid.Executor(fluid.CPUPlace())
        dataset.set_thread(args.num_thread)
    exe.run(fluid.default_startup_program())

    for epoch_id in range(args.num_epoch):
        start = time.time()
        sys.stderr.write('\nepoch%d start ...\n' % (epoch_id + 1))
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=dataset,
            fetch_list=[loss, auc],
            fetch_info=['loss', 'auc'],
            debug=False,
            print_period=args.print_steps)
        model_dir = os.path.join(args.model_output_dir,
                                 'epoch_' + str(epoch_id + 1))
        sys.stderr.write('epoch%d is finished and takes %f s\n' % (
            (epoch_id + 1), time.time() - start))
        fluid.io.save_persistables(
            executor=exe,
            dirname=model_dir,
            main_program=fluid.default_main_program())


if __name__ == '__main__':
    utils.check_version()
    train()
