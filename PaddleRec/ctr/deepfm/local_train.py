from args import parse_args
import os
import paddle.fluid as fluid
import sys
from network_conf import ctr_deepfm_model
import time
import numpy
import pickle


def train():
    args = parse_args()
    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    loss, auc, data_list = ctr_deepfm_model(args.embedding_size, args.num_field,
                                            args.num_feat, args.layer_sizes,
                                            args.act, args.reg)
    optimizer = fluid.optimizer.SGD(
        learning_rate=args.lr,
        regularization=fluid.regularizer.L2DecayRegularizer(args.reg))
    optimizer.minimize(loss)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(data_list)
    pipe_command = 'python criteo_reader.py'
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(args.batch_size)
    dataset.set_thread(args.num_thread)

    whole_filelist = [
        'data/raw_data/part-%d' % x
        for x in range(len(os.listdir('data/raw_data')))
    ]
    train_file_idx = pickle.load(
        open('data/aid_data/train_file_idx.pkl2', 'rb'))
    train_filelist = [whole_filelist[idx] for idx in train_file_idx]

    for epoch_id in range(args.num_epoch):
        start = time.time()
        dataset.set_filelist(train_filelist)
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=dataset,
            fetch_list=[loss, auc],
            fetch_info=['loss', 'auc'],
            debug=False,
            print_period=1000)
        model_dir = args.model_output_dir + '/epoch_' + str(epoch_id + 1)
        sys.stderr.write('epoch%d is finished and takes %f s\n' % (
            (epoch_id + 1), time.time() - start))
        fluid.io.save_persistables(
            executor=exe,
            dirname=model_dir,
            main_program=fluid.default_main_program())


if __name__ == '__main__':
    train()
