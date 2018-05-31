import numpy as np
import argparse
import paddle.fluid as fluid
from utils import read_data_sets
from paddle.fluid.layers import log, elementwise_max, create_tensor, assign, reduce_sum


def mlp_teacher(img, drop_prob):
    h1 = fluid.layers.fc(input=img, size=1200, act='relu')
    drop1 = fluid.layers.dropout(h1, dropout_prob=drop_prob)
    h2 = fluid.layers.fc(input=drop1, size=1200, act='relu')
    drop2 = fluid.layers.dropout(h2, dropout_prob=drop_prob)
    logits = fluid.layers.fc(input=drop2, size=10, act=None)
    return logits


def mlp_student(img, drop_prob, h_size):
    h1 = fluid.layers.fc(input=img, size=h_size, act='relu')
    drop1 = fluid.layers.dropout(h1, dropout_prob=drop_prob)
    h2 = fluid.layers.fc(input=drop1, size=h_size, act='relu')
    drop2 = fluid.layers.dropout(h2, dropout_prob=drop_prob)
    logits = fluid.layers.fc(input=drop2, size=10, act=None)
    return logits


def softmax_with_temperature(logits, temp=1.0):
    logits_with_temp = logits / temp
    _softmax = fluid.layers.softmax(logits_with_temp)
    return _softmax


def soft_crossentropy(input, label):
    '''
    Since the Paddlepaddle crossentroy have numerical overflow problems, so I reimplement the operation
    :param input: softmax probs
    :param label: soft labels
    :return:
    '''
    epsilon = 1e-8
    eps = fluid.layers.ones(shape=[1], dtype='float32') * epsilon
    loss = reduce_sum(
        -1.0 * label * log(elementwise_max(input, eps)), dim=1, keep_dim=True)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        type=str,
        default="teacher",
        help="choose from teacher or student")
    parser.add_argument(
        "--stu_hsize",
        type=int,
        default=30,
        help="The hidden layer size of student net")
    parser.add_argument(
        "--drop_prob",
        type=float,
        default=0.1,
        help="The dropout probability of fully connected layers")
    parser.add_argument(
        "--temp",
        type=float,
        default=4.0,
        help='The temperature of softmax which is used to generate soft targets')
    parser.add_argument(
        "--teacher_dir",
        type=str,
        default="./models/teacher_net",
        help="Set the directory for saving teacher network parameters")
    parser.add_argument(
        "--student_dir",
        type=str,
        default="./models/student_net",
        help="set the directory for saving student network parameters")
    parser.add_argument(
        "--use_soft",
        type=bool,
        default=False,
        help="Whether using soft targets to train student network, set a switch to True"
    )
    parser.add_argument(
        "--epoch_num", type=int, default=200, help="Number of training epoches")

    args = parser.parse_args()
    print(args)

    if args.phase == 'teacher':
        print("Training Teacher network")
        drop_prob = args.drop_prob
        teacher_dir = args.teacher_dir
        epoch_num = args.epoch_num

        img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        # construct the computational graph
        logits = mlp_teacher(img, drop_prob=drop_prob)
        softmax_logits = fluid.layers.softmax(logits)
        cost = fluid.layers.cross_entropy(input=softmax_logits, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        # optimizer = fluid.optimizer.SGD(learning_rate=0.001, regularization=fluid.regularizer.L2Decay(1e-5))
        optimizer.minimize(avg_cost)
        batch_acc = fluid.layers.accuracy(input=softmax_logits, label=label)

        train_set, test_set = read_data_sets(
            is_soft=False, one_hot=False, reshape=True)
        G_batch_size = 100
        T_batch_size = 100
        Tr_batch_size = 128
        train_x = train_set.images
        train_y = train_set.labels
        test_x = test_set.images
        test_y = test_set.labels
        print(train_set.num_examples, train_set.images.shape)
        train_iternum = train_set.num_examples // Tr_batch_size
        test_iternum = test_set.num_examples // T_batch_size

        place = fluid.CUDAPlace(2)
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
        exe.run(fluid.default_startup_program())
        inference_program = fluid.default_main_program().clone(for_test=False)

        print("Begin training teacher network")
        for epoch in range(epoch_num):
            train_loss_list = []
            train_acc_list = []
            for i in range(train_iternum):
                train_batch = train_set.next_batch(Tr_batch_size)
                loss, acc = exe.run(fluid.default_main_program(),
                                    feed=feeder.feed(train_batch),
                                    fetch_list=[avg_cost, batch_acc])
                train_loss_list.append(loss)
                train_acc_list.append(acc)

            test_loss_list = []
            test_acc_list = []
            for i in range(test_iternum):
                test_batch = list(
                    zip(test_x[i * T_batch_size:(i + 1) * T_batch_size], test_y[
                        i * T_batch_size:(i + 1) * T_batch_size]))
                loss, acc = exe.run(inference_program,
                                    feed=feeder.feed(test_batch),
                                    fetch_list=[avg_cost, batch_acc])
                test_loss_list.append(loss)
                test_acc_list.append(acc)

            print(
                "Epoch {}, train acc {}, train loss {} ; test acc {}, test loss {} ".
                format(epoch,
                       np.mean(train_acc_list),
                       np.mean(train_loss_list),
                       np.mean(test_acc_list), np.mean(test_loss_list)))
        fluid.io.save_params(
            exe, dirname=teacher_dir, main_program=fluid.default_main_program())
        print('Train teacher network done')
    elif args.phase == 'student':
        print("Training Student Network")
        drop_prob = args.drop_prob
        student_dir = args.student_dir
        h_size = args.stu_hsize
        temp = args.temp
        use_soft = args.use_soft
        epoch_num = args.epoch_num

        img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[10], dtype='float32')
        soft_label = fluid.layers.data(
            name='soft_label', shape=[10], dtype='float32')
        logits = mlp_student(img, h_size=h_size, drop_prob=drop_prob)
        softmax_logits = fluid.layers.softmax(logits)
        temp_softmax_logits = softmax_with_temperature(logits, temp=temp)
        hard_loss = soft_crossentropy(input=softmax_logits, label=label)
        soft_loss = soft_crossentropy(
            input=temp_softmax_logits, label=soft_label)
        if use_soft:
            use_soft = 'T'
            cost = hard_loss + soft_loss * temp**2
        else:
            use_soft = 'F'
            cost = hard_loss
        avg_cost = fluid.layers.mean(x=cost)
        optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        # optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        optimizer.minimize(avg_cost)

        top1_values, top1_indices = fluid.layers.topk(label, k=1)
        print(top1_indices.shape)
        batch_acc = fluid.layers.accuracy(
            input=softmax_logits, label=top1_indices)

        dirname = student_dir + '_h_{}_t_{}_soft_{}'.format(
            h_size, str(temp), use_soft)  # 'F' means soft loss false
        inference_program = fluid.default_main_program().clone(for_test=False)
        place = fluid.CUDAPlace(1)
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(
            feed_list=[img, label, soft_label], place=place)
        exe.run(fluid.default_startup_program())

        train_set, test_set = read_data_sets(
            is_soft=True, one_hot=True, reshape=True, temp=str(temp))
        T_batch_size = 100
        Tr_batch_size = 128
        train_x = train_set.images
        train_y = train_set.labels
        test_x = test_set.images
        test_y = test_set.labels
        test_soft_y = test_set.soft_labels
        train_iternum = train_set.num_examples // Tr_batch_size
        test_iternum = test_set.num_examples // T_batch_size
        for epoch in range(epoch_num):
            train_loss_list = []
            train_acc_list = []
            for i in range(train_iternum):
                train_batch = train_set.next_batch(Tr_batch_size)
                loss, acc = exe.run(fluid.default_main_program(),
                                    feed=feeder.feed(train_batch),
                                    fetch_list=[avg_cost, batch_acc])
                train_loss_list.append(loss)
                train_acc_list.append(acc)

            test_loss_list = []
            test_acc_list = []
            for i in range(test_iternum):
                test_batch = list(
                    zip(test_x[i * T_batch_size:(i + 1) * T_batch_size], test_y[
                        i * T_batch_size:(i + 1) * T_batch_size], test_soft_y[
                            i * T_batch_size:(i + 1) * T_batch_size]))
                loss, acc = exe.run(inference_program,
                                    feed=feeder.feed(test_batch),
                                    fetch_list=[avg_cost, batch_acc])
                test_loss_list.append(loss)
                test_acc_list.append(acc)
            print(
                "Epoch {}, train acc {}, train loss {} ; test acc {}, test loss {} ".
                format(epoch,
                       np.mean(train_acc_list),
                       np.mean(train_loss_list),
                       np.mean(test_acc_list), np.mean(test_loss_list)))

        fluid.io.save_params(
            exe, dirname=dirname, main_program=fluid.default_main_program())
        print('Train Student done')
        print(dirname)

    else:
        print("Please choose teacher or student for --phase")


if __name__ == '__main__':
    main()
