import net
import numpy as np
import paddle.fluid as fluid


def gen_data(dict_dim=100, class_size=2, batch_size=32, max_len=10):
    return {
        "input": np.random.randint(
            dict_dim, size=(batch_size, max_len)).astype('int64'),
        "seq_len": np.random.randint(
            1, high=max_len, size=(batch_size)).astype('int64'),
        "label": np.random.randint(
            class_size, size=(batch_size, 1)).astype('int64')
    }


main_program = fluid.default_startup_program()
startup_program = fluid.default_main_program()
dict_dim = 100
with fluid.program_guard(main_program, startup_program):
    cost = net.cnn_net(dict_dim=dict_dim)
    optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    optimizer.minimize(cost)
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(startup_program)
    step = 100
    for i in range(step):
        cost_val = exe.run(main_program,
                           feed=gen_data(),
                           fetch_list=[cost.name])
        print("step%d cost=%f" % (i, cost_val[0]))
