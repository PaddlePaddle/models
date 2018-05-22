import paddle
import paddle.fluid as fluid
import numpy as np


def load_vars():
    vars = {}
    name_map = {}
    with open('./vgg/names.map', 'r') as map_file:
        for param in map_file:
            fd_name, cf_name = param.strip().split('\t')
            name_map[fd_name] = cf_name

    cf_vars = np.load('./vgg/vgg.npy').item()
    for fd_name in name_map:
        cf_name = name_map[fd_name]
        cf_var_weights = cf_vars[cf_name]['weights']
        cf_var_biases = cf_vars[cf_name]['biases']
        vars[fd_name + '.w_0'] = cf_var_weights
        vars[fd_name + '.b_0'] = cf_var_biases
    return vars


def load_and_set_vars(place):
    vars = load_vars()
    for k, v in vars.items():
        t = fluid.global_scope().find_var(k).get_tensor()
        assert np.array(t).shape == v.shape
        t.set(v, place)


if __name__ == "__main__":
    load_vars()
