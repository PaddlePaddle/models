import paddle.fluid as fluid
import numpy as np
import time
from args import *


def fc_layers(input, layers, acts, prefix):
    fc_layers_input = [input]
    fc_layers_size = layers
    fc_layers_act = acts
    init_range = 0.2
    scales_tmp = [input.shape[1]] + fc_layers_size
    scales = []
    for i in range(len(scales_tmp)):
        scales.append(init_range / (scales_tmp[i]**0.5))
    for i in range(len(fc_layers_size)):
        name = prefix + "_" + str(i)
        fc = fluid.layers.fc(
                input = fc_layers_input[-1],
                size = fc_layers_size[i],
                act = fc_layers_act[i],
                param_attr = \
                        fluid.ParamAttr(learning_rate=1.0, \
                        initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0 * scales[i])),
                bias_attr = \
                        fluid.ParamAttr(learning_rate=1.0, \
                        initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0 * scales[i])),
                name=name)
        fc_layers_input.append(fc)
    return fc_layers_input[-1]


def mmoe_layer(inputs, expert_num=8, gate_num=3):

    expert_out = []
    expert_nn = [3]
    expert_act = ['relu']
    for i in range(0, expert_num):
        cur_expert = fc_layers(inputs, expert_nn, expert_act,
                               'expert_' + str(i))
        expert_out.append(cur_expert)
    expert_concat = fluid.layers.concat(expert_out, axis=1)
    expert_concat = fluid.layers.reshape(expert_concat,
                                         [-1, expert_num, expert_nn[-1]])

    outs = []
    for i in range(0, gate_num):
        cur_gate = fluid.layers.fc(input=inputs,
                                   size=expert_num,
                                   act='softmax',
                                   name='gate_' + str(i))
        cur_gate_expert = fluid.layers.elementwise_mul(
            expert_concat, cur_gate, axis=0)
        cur_gate_expert = fluid.layers.reduce_sum(cur_gate_expert, dim=1)
        cur_fc = fc_layers(cur_gate_expert, [64, 32, 16, 1],
                           ['relu', 'relu', 'relu', None], 'out_' + str(i))
        outs.append(cur_fc)
    return outs


def model(dict_dim, emb_dim):
    label_like = fluid.data(
        name="label_like", shape=[-1, 1], dtype="int64", lod_level=0)
    label_comment = fluid.data(
        name="label_comment", shape=[-1, 1], dtype="int64", lod_level=0)
    label_share = fluid.data(
        name="label_share", shape=[-1, 1], dtype="int64", lod_level=0)

    a_data = fluid.data(name="a", shape=[-1, 1], dtype="int64")
    emb = fluid.layers.embedding(input=a_data, size=[dict_dim, emb_dim])

    outs = mmoe_layer(emb, expert_num=8, gate_num=3)

    output_like = fluid.layers.sigmoid(
        fluid.layers.clip(
            outs[0], min=-15.0, max=15.0), name="output_like")
    output_comment = fluid.layers.sigmoid(
        fluid.layers.clip(
            outs[1], min=-15.0, max=15.0), name="output_comment")
    output_share = fluid.layers.sigmoid(
        fluid.layers.clip(
            outs[2], min=-15.0, max=15.0), name="output_share")

    cost_like = fluid.layers.log_loss(
        input=output_like,
        label=fluid.layers.cast(
            x=label_like, dtype='float32'))
    cost_comment = fluid.layers.log_loss(
        input=output_comment,
        label=fluid.layers.cast(
            x=label_comment, dtype='float32'))
    cost_share = fluid.layers.log_loss(
        input=output_share,
        label=fluid.layers.cast(
            x=label_share, dtype='float32'))

    avg_cost_like = fluid.layers.mean(x=cost_like)
    avg_cost_comment = fluid.layers.mean(x=cost_comment)
    avg_cost_share = fluid.layers.mean(x=cost_share)

    cost = avg_cost_like + avg_cost_comment + avg_cost_share
    return cost, [a_data, label_like, label_comment, label_share]


args = parse_args()
batch_size = args.batch_size
dict_dim = args.dict_dim
emb_dim = args.emb_dim

print("batch_size:[%d], dict_dim:[%d], emb_dim:[%d], learning_rate:[%.4f]" %
      (batch_size, dict_dim, emb_dim, args.base_lr))

loss, data_list = model(dict_dim, emb_dim)
sgd = fluid.optimizer.SGD(learning_rate=args.base_lr)
sgd.minimize(loss)
place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
for batch_id in range(100):
    data = [
        np.random.randint(
            2, size=(batch_size, 1)).astype('int64') for i in range(4)
    ]
    begin = time.time()
    loss_data, = exe.run(fluid.default_main_program(),
                         feed={
                             "a": data[0],
                             "label_like": data[1],
                             "label_comment": data[2],
                             "label_share": data[3]
                         },
                         fetch_list=[loss.name])
    end = time.time()
    print("batch_id:[%d], loss:[%.5f], batch_time:[%.5f s]" %
          (batch_id, float(np.array(loss_data)), end - begin))
