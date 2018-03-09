import sys
import numpy as np
import ast


def load_parameter(file_name):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header.
        return np.fromfile(f, dtype=np.float32)


def load_param(name_map_file, old_param_dir):
    result = {}
    name_map = {}
    shape_map = {}
    with open(name_map_file, 'r') as map_file:
        for param in map_file:
            old_name, new_name, shape = param.strip().split('=')
            name_map[new_name] = old_name
            shape_map[new_name] = ast.literal_eval(shape)

    for new_name in name_map:
        result[new_name] = load_parameter("/".join(
            [old_param_dir, name_map[new_name]])).reshape(shape_map[new_name])
    return result


if __name__ == "__main__":
    name_map_file = "./name.map"
    old_param_dir = "./data/model/results/pass-00062/"
    result = load_param(name_map_file, old_param_dir)
    for p in result:
        print "name: %s; param.shape: %s" % (p, result[p].shape)
