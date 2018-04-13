"""
For http://wiki.baidu.com/display/LegoNet/Text+Classification
"""
import paddle.fluid as fluid
import paddle.v2 as paddle
import numpy as np
import sys
import time
import unittest
import contextlib
import utils


def bow_net(data, label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
    """
    bow net
    """
    emb = fluid.layers.embedding(input=data, 
                                size=[dict_dim, emb_dim])
    bow = fluid.layers.sequence_pool(
        input=emb,
        pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh,
                        size=hid_dim, act = "tanh")
    fc_2 = fluid.layers.fc(input=fc_1,
                        size=hid_dim2, act = "tanh")
    prediction = fluid.layers.fc(input=[fc_2],
                             size=class_dim,
                             act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    
    return avg_cost, acc, prediction


def conv_net(data, label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            win_size=3):
    """
    conv net
    """
    emb = fluid.layers.embedding(input=data, 
                                size=[dict_dim, emb_dim])

    conv_3 = fluid.nets.sequence_conv_pool(input=emb,
                                    num_filters=hid_dim,
                                    filter_size=win_size,
                                    act="tanh",
                                    pool_type="max")

    fc_1 = fluid.layers.fc(input=[conv_3],
                                    size=hid_dim2)

    prediction = fluid.layers.fc(input=[fc_1],
                             size=class_dim,
                             act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    
    return avg_cost, acc, prediction


def lstm_net(data, label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
    """
    lstm net
    """
    emb = fluid.layers.embedding(input=data, 
                                size=[dict_dim, emb_dim])

    fc0 = fluid.layers.fc(input=emb, 
                        size=hid_dim * 4, 
                        act='tanh')

    lstm_h, c = fluid.layers.dynamic_lstm(input=fc0, 
                        size=hid_dim * 4, 
                        is_reverse=False)

    lstm_max = fluid.layers.sequence_pool(input=lstm_h, 
                        pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max)

    fc1 = fluid.layers.fc(input=lstm_max_tanh, 
                        size=hid_dim2, 
                        act='tanh')

    prediction = fluid.layers.fc(input=fc1, 
                        size=class_dim, 
                        act='softmax')

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    
    return avg_cost, acc, prediction


def gru_net(data, label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
    """
    gru net
    """
    emb = fluid.layers.embedding(input=data, 
                                size=[dict_dim, emb_dim])

    fc0 = fluid.layers.fc(input=emb, 
                        size=hid_dim * 3)

    gru_h = fluid.layers.dynamic_gru(input=fc0, 
                        size=hid_dim, 
                        is_reverse=False)

    gru_max = fluid.layers.sequence_pool(input=gru_h, 
                        pool_type='max')
    gru_max_tanh = fluid.layers.tanh(gru_max)

    fc1 = fluid.layers.fc(input=gru_max_tanh, 
                        size=hid_dim2, 
                        act='tanh')

    prediction = fluid.layers.fc(input=fc1, 
                        size=class_dim, 
                        act='softmax')

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    
    return avg_cost, acc, prediction


def train(train_reader,
        word_dict,
        network,
        use_cuda,
        parallel,
        save_dirname,
        lr=0.2,
        batch_size=128,
        pass_num=30):
    """
    train network
    """
    data = fluid.layers.data(
        name="words", 
        shape=[1], 
        dtype="int64", 
        lod_level=1)

    label = fluid.layers.data(
        name="label", 
        shape=[1], 
        dtype="int64")

    if not parallel:
        cost, acc, prediction = network(
            data, label, len(word_dict))
    else:
        places = fluid.layers.get_places(device_count = 2)
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            cost, acc, prediction = network(
            pd.read_input(data), 
            pd.read_input(label), 
            len(word_dict))

            pd.write_output(cost)
            pd.write_output(acc)

        cost, acc = pd()
        cost = fluid.layers.mean(cost)
        acc = fluid.layers.mean(acc)

    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    sgd_optimizer.minimize(cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    exe.run(fluid.default_startup_program())
    for pass_id in xrange(pass_num):
        avg_cost_list, avg_acc_list = [], []
        for data in train_reader():
            avg_cost_np, avg_acc_np = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[cost, acc])
            avg_cost_list.append(avg_cost_np)
            avg_acc_list.append(avg_acc_np)
        print("pass_id: %d, avg_acc: %f" % (pass_id, np.mean(avg_acc_list)))
    # save_model
    fluid.io.save_inference_model(
            save_dirname, 
            ["words", "label"],
            acc, exe)


def test(test_reader, use_cuda, 
        save_dirname=None):
    """
    test function
    """
    if save_dirname is None:
        print(str(save_dirname) + " cannot be found")
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
        fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

        total_acc = 0.0
        total_count = 0
        for data in test_reader():
            acc = exe.run(inference_program,
                    feed = utils.data2tensor(data, place),
                    fetch_list=fetch_targets,
                    return_numpy=True)
            total_acc += acc[0] * len(data)
            total_count += len(data)
        print("test_acc: %f" % (total_acc / total_count))


def main(network,
        dataset,
        model_conf,
        use_cuda,
        parallel,
        batch_size,
        lr=0.2,
        pass_num=30):
    """
    main function
    """
    word_dict, train_reader, test_reader = utils.prepare_data(
            dataset, self_dict = False,
            batch_size = batch_size, buf_size = 50000)

    train(train_reader, word_dict,
            network, use_cuda=use_cuda,
            parallel = parallel,
            save_dirname=model_conf,
            lr=lr,
            pass_num=pass_num, 
            batch_size=batch_size)

    test(test_reader, use_cuda=use_cuda,
            save_dirname=model_conf)


class TestModel(unittest.TestCase):
    """
    Test Case Module
    """

    @contextlib.contextmanager
    def new_program_scope(self):
        """
        setting external env
        """
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield
    
    @unittest.skip(reason='success, total time:14.19s')
    def test_bow_cpu(self):
        """
        Test bow cpu single thread
        """
        with self.new_program_scope():
            main(bow_net, "tiny_imdb",
                "bow.cpu", False, False, 128)

    @unittest.skip(reason='success, total time:7.62s')
    def test_bow_gpu(self):
        """
        Test bow gpu single thread
        """
        with self.new_program_scope():
            main(bow_net, "tiny_imdb",
                "bow.gpu", True, False, 128)

    @unittest.skip(reason='success, total time:15.02s')
    def test_bow_cpu_mthread(self):
        """
        Test bow cpu mthread
        """
        with self.new_program_scope():
            main(bow_net, "tiny_imdb",
                "bow.cpu_mthread", False, True, 128)

    @unittest.skip(reason='success, total time:9.45s')
    def test_bow_gpu_mthread(self):
        """
        Test bow gpu mthread
        """
        with self.new_program_scope():
            main(bow_net, "tiny_imdb",
                "bow.gpu_mthread", True, True, 128)

    @unittest.skip(reason='success, total time:85.0s')
    def test_cnn_cpu(self):
        """
        Test cnn cpu single thread
        """
        with self.new_program_scope():
            main(conv_net, "tiny_imdb",
                "conv.cpu", False, False, 128)

    @unittest.skip(reason='success, total time:12.0s')
    def test_cnn_gpu(self):
        """
        Test cnn gpu single thread
        """
        with self.new_program_scope():
            main(conv_net, "tiny_imdb",
                "conv.gpu", True, False, 128)

    @unittest.skip(reason='success, total time:53.0s')
    def test_cnn_cpu_mthread(self):
        """
        Test cnn cpu mthread
        """
        with self.new_program_scope():
            main(conv_net, "tiny_imdb",
                "conv.cpu_mthread", False, True, 128)

    @unittest.skip(reason='success, total time:10.9s')
    def test_cnn_gpu_mthread(self):
        """
        Test cnn gpu mthread
        """
        with self.new_program_scope():
            main(conv_net, "tiny_imdb",
                "conv.gpu_mthread", True, True, 128)

    @unittest.skip(reason='success, total time:232.5s')
    def test_lstm_cpu(self):
        """
        Test lstm cpu single thread
        """
        with self.new_program_scope():
            main(lstm_net, "tiny_imdb",
                "lstm.cpu", False, False, 128)

    @unittest.skip(reason='success, total time:26.5s')
    def test_lstm_gpu(self):
        """
        Test lstm gpu single thread
        """
        with self.new_program_scope():
            main(lstm_net, "tiny_imdb",
                "lstm.gpu", True, False, 128)

    @unittest.skip(reason='success, total time:135.0s')
    def test_lstm_cpu_mthread(self):
        """
        Test lstm cpu mthread
        """
        with self.new_program_scope():
            main(lstm_net, "tiny_imdb",
                "lstm.cpu_mthread", False, True, 128)

    @unittest.skip(reason='success, total time:26.23s')
    def test_lstm_gpu_mthread(self):
        """
        Test lstm gpu mthread
        """
        with self.new_program_scope():
            main(lstm_net, "tiny_imdb",
                "lstm.gpu_mthread", True, True, 128)

    @unittest.skip(reason='success, total time:163.0s')
    def test_gru_cpu(self):
        """
        Test gru cpu single thread
        """
        with self.new_program_scope():
            main(gru_net, "tiny_imdb",
                "gru.cpu", False, False, 128)

    @unittest.skip(reason='success, total time:28.88s')
    def test_gru_gpu(self):
        """
        Test gru gpu single thread
        """
        with self.new_program_scope():
            main(gru_net, "tiny_imdb",
                "gru.gpu", True, False, 128, 0.02, 30)

    @unittest.skip(reason='success, total time:97.15s')
    def test_gru_cpu_mthread(self):
        """
        Test gru cpu mthread
        """
        with self.new_program_scope():
            main(gru_net, "tiny_imdb",
                "gru.cpu_mthread", False, True, 128)

    @unittest.skip(reason='success, total time:26.05s')
    def test_gru_gpu_mthread(self):
        """
        Test gru gpu mthread
        """
        with self.new_program_scope():
            main(gru_net, "tiny_imdb",
                "gru.gpu_mthread", True, True, 128)

if __name__ == "__main__":
    unittest.main()
