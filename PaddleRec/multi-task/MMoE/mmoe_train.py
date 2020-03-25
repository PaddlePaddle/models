import paddle.fluid as fluid
import pandas as pd
import numpy as np
import paddle
import time
import utils
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from args import *
import warnings
warnings.filterwarnings("ignore")
#显示所有列
pd.set_option('display.max_columns', None)


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


def MMOE(expert_num=8, gate_num=2):
    a_data = fluid.data(name="a", shape=[-1, 499], dtype="float32")
    label_income = fluid.data(
        name="label_income", shape=[-1, 2], dtype="float32", lod_level=0)
    label_marital = fluid.data(
        name="label_marital", shape=[-1, 2], dtype="float32", lod_level=0)

    # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
    expert_outputs = []
    for i in range(0, expert_num):
        expert_output = fluid.layers.fc(
            input=a_data,
            size=4,
            act='relu',
            bias_attr=fluid.ParamAttr(learning_rate=1.0),
            name='expert_' + str(i))
        expert_outputs.append(expert_output)
    expert_concat = fluid.layers.concat(expert_outputs, axis=1)
    expert_concat = fluid.layers.reshape(expert_concat, [-1, 4, 8])

    # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
    gate_outputs = []
    for i in range(0, gate_num):
        cur_gate = fluid.layers.fc(input=a_data,
                                   size=8,
                                   act='softmax',
                                   bias_attr=fluid.ParamAttr(learning_rate=1.0),
                                   name='gate_' + str(i))
        gate_outputs.append(cur_gate)

    # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
    final_outputs = []
    for gate_output in gate_outputs:
        expanded_gate_output = fluid.layers.reshape(gate_output, [-1, 1, 8])
        weighted_expert_output = expert_concat * fluid.layers.expand(
            expanded_gate_output, expand_times=[1, 4, 1])
        final_outputs.append(
            fluid.layers.reduce_sum(
                weighted_expert_output, dim=2))

    # Build tower layer from MMoE layer
    output_layers = []
    for index, task_layer in enumerate(final_outputs):
        tower_layer = fluid.layers.fc(input=final_outputs[index],
                                      size=8,
                                      act='relu',
                                      name='task_layer_' + str(index))
        output_layer = fluid.layers.fc(input=tower_layer,
                                       size=2,
                                       act='softmax',
                                       name='output_layer_' + str(index))
        output_layers.append(output_layer)

    cost_income = paddle.fluid.layers.cross_entropy(
        input=output_layers[0], label=label_income, soft_label=True)
    cost_marital = paddle.fluid.layers.cross_entropy(
        input=output_layers[1], label=label_marital, soft_label=True)

    label_income_1 = fluid.layers.slice(
        label_income, axes=[1], starts=[1], ends=[2])
    label_marital_1 = fluid.layers.slice(
        label_marital, axes=[1], starts=[1], ends=[2])

    auc_income, batch_auc_1, auc_states_1 = fluid.layers.auc(
        input=output_layers[0],
        label=fluid.layers.cast(
            x=label_income_1, dtype='int64'))
    auc_marital, batch_auc_2, auc_states_2 = fluid.layers.auc(
        input=output_layers[1],
        label=fluid.layers.cast(
            x=label_marital_1, dtype='int64'))

    avg_cost_income = fluid.layers.mean(x=cost_income)
    avg_cost_marital = fluid.layers.mean(x=cost_marital)

    cost = avg_cost_income + avg_cost_marital

    return [
        a_data, label_income, label_marital
    ], cost, output_layers[0], output_layers[
        1], label_income, label_marital, auc_income, auc_marital, auc_states_1, auc_states_2


args = parse_args()
train_path = args.train_data_path
test_path = args.test_data_path
batch_size = args.batch_size
expert_num = args.expert_num
epochs = args.epochs
gate_num = args.gate_num

print("batch_size:[%d],expert_num:[%d],gate_num[%d]" %
      (batch_size, expert_num, gate_num))

train_reader = utils.prepare_reader(train_path, batch_size)
test_reader = utils.prepare_reader(test_path, batch_size)
#for data in train_reader():
#    print(data[0][1])    

data_list, loss, out_1, out_2, label_1, label_2, auc_income, auc_marital, auc_states_1, auc_states_2 = MMOE(
    expert_num, gate_num)

Adam = fluid.optimizer.AdamOptimizer()
Adam.minimize(loss)
place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
test_program = fluid.default_main_program().clone(for_test=True)

loader = fluid.io.DataLoader.from_generator(
    feed_list=data_list, capacity=batch_size, iterable=True)
loader.set_sample_list_generator(train_reader, places=place)

test_loader = fluid.io.DataLoader.from_generator(
    feed_list=data_list, capacity=batch_size, iterable=True)
test_loader.set_sample_list_generator(test_reader, places=place)
mean_auc_income = []
mean_auc_marital = []
inference_scope = fluid.Scope()

for epoch in range(epochs):
    for var in auc_states_1:  # reset auc states
        set_zero(var.name, place=place)
    for var in auc_states_2:  # reset auc states
        set_zero(var.name, place=place)
    begin = time.time()
    auc_1_p = 0.0
    auc_2_p = 0.0
    loss_data = 0.0
    for batch_id, train_data in enumerate(loader()):

        loss_data, out_income, out_marital, label_income, label_marital, auc_1_p, auc_2_p = exe.run(
            feed=train_data,
            fetch_list=[
                loss.name, out_1, out_2, label_1, label_2, auc_income,
                auc_marital
            ],
            return_numpy=True)

    for var in auc_states_1:  # reset auc states
        set_zero(var.name, place=place)
    for var in auc_states_2:  # reset auc states
        set_zero(var.name, place=place)
    test_auc_1_p = 0.0
    test_auc_2_p = 0.0
    for batch_id, test_data in enumerate(test_loader()):

        test_out_income, test_out_marital, test_label_income, test_label_marital, test_auc_1_p, test_auc_2_p = exe.run(
            program=test_program,
            feed=test_data,
            fetch_list=[
                out_1, out_2, label_1, label_2, auc_income, auc_marital
            ],
            return_numpy=True)
    mean_auc_income.append(test_auc_1_p)
    mean_auc_marital.append(test_auc_2_p)
    end = time.time()
    print(
        "epoch_id:[%d],epoch_time:[%.5f s],loss:[%.5f],train_auc_income:[%.5f],train_auc_marital:[%.5f],test_auc_income:[%.5f],test_auc_marital:[%.5f]"
        % (epoch, end - begin, loss_data, auc_1_p, auc_2_p, test_auc_1_p,
           test_auc_2_p))
print("mean_auc_income:[%.5f],mean_auc_marital[%.5f]" %
      (np.mean(mean_auc_income), np.mean(mean_auc_marital)))
