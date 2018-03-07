import paddle.v2 as paddle
import paddle.fluid as fluid
import numpy as np


def load_vars():
    vars = {}
    name_map = {}
    with open('./ssd_mobilenet_v1_coco/names.map', 'r') as map_file:
        for param in map_file:
            fd_name, tf_name = param.strip().split('\t')
            name_map[fd_name] = tf_name

    tf_vars = np.load(
        './ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2017_11_17.npy').item()
    for fd_name in name_map:
        tf_name = name_map[fd_name]
        tf_var = tf_vars[tf_name]
        if len(tf_var.shape) == 4 and 'depthwise' in tf_name:
            vars[fd_name] = np.transpose(tf_var, (2, 3, 0, 1))
        elif len(tf_var.shape) == 4:
            vars[fd_name] = np.transpose(tf_var, (3, 2, 0, 1))
        else:
            vars[fd_name] = tf_var

    return vars


def load_and_set_vars(place):
    vars = load_vars()
    for k, v in vars.items():
        t = fluid.global_scope().find_var(k).get_tensor()
        #print(np.array(t).shape, v.shape, k)
        assert np.array(t).shape == v.shape
        t.set(v, place)


if __name__ == "__main__":
    load_vars()
