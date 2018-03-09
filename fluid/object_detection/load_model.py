import paddle.v2 as paddle
import paddle.fluid as fluid
import numpy as np


# From npy
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


# From Paddle V1
def load_paddlev1_vars():
    vars = {}
    name_map = {}
    with open('./caffe2paddle/names.map', 'r') as map_file:
        for param in map_file:
            fd_name, tf_name = param.strip().split('\t')
            name_map[fd_name] = tf_name

    from operator import mul

    def load(file_name, shape):
        with open(file_name, 'rb') as f:
            f.read(16)
            arr = np.fromfile(f, dtype=np.float32)
            assert arr.size == reduce(mul, shape)
            return arr.reshape(shape)

    for fd_name in name_map:
        v1_name = name_map[fd_name]
        t = fluid.global_scope().find_var(fd_name).get_tensor()
        shape = np.array(t).shape
        v1_var = load('./caffe2paddle/' + v1_name, shape)
        t.set(v1_var, place)


if __name__ == "__main__":
    load_vars()
