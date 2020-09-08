"""
This code is based on https://github.com/garrickbrazil/M3D-RPN/blob/master/lib/core.py
This file is meant to contain all functions of the detective framework
which are "specific" to the framework but generic among experiments.

For example, all the experiments need to initialize configs, training models,
log stats, display stats, and etc. However, these functions are generally fixed
to this framework and cannot be easily transferred in other projects.
"""

# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from shapely.geometry import Polygon
#import matplotlib.pyplot as plt
from copy import copy
import importlib
import random
#import visdom
#import torch
import paddle.fluid as fluid
import paddle
import shutil
import sys
import os
import cv2
import math
import numpy as np
import struct
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.proto.framework_pb2 import VarType
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.util import *


def init_config(conf_name):
    """
    Loads configuration file, by checking for the conf_name.py configuration file as
    ./config/<conf_name>.py which must have function "Config".

    This function must return a configuration dictionary with any necessary variables for the experiment.
    """

    conf = importlib.import_module('config.' + conf_name).Config()

    return conf


import paddle.fluid as fluid


class MyPolynomialDecay(fluid.dygraph.PolynomialDecay):
    def step(self):
        tmp_step_num = self.step_num
        tmp_decay_steps = self.decay_steps
        tmp_step_num = self.create_lr_var(tmp_step_num if tmp_step_num < self.
                                          decay_steps else self.decay_steps)

        scale = float(tmp_decay_steps) / (
            1 - float(self.end_learning_rate / self.learning_rate)**
            (1 / self.power))

        decay_lr = self.learning_rate * ((1 - float(tmp_step_num) / scale)
                                         **self.power)

        return decay_lr


def adjust_lr(conf):

    #if 'batch_skip' in conf and ((iter + 1) % conf.batch_skip) > 0: return

    if conf.solver_type.lower() == 'sgd':

        lr = conf.lr
        lr_steps = conf.lr_steps
        max_iter = conf.max_iter
        lr_policy = conf.lr_policy
        lr_target = conf.lr_target

        # perform the exact number of steps needed to get to lr_target
        # if lr_policy.lower() == 'step':
        #     scale = (lr_target / lr) ** (1 / total_steps)
        #     lr *= scale ** step_count

        # compute the scale needed to go from lr --> lr_target
        # using a polynomial function instead.
        if lr_policy.lower() == 'poly':

            lr = MyPolynomialDecay(lr, max_iter, lr_target, power=0.9)

        else:
            raise ValueError('{} lr_policy not understood'.format(lr_policy))

    return lr


def init_training_model(conf, backbone, cache_folder):
    """
    This function is meant to load the training model and optimizer, which expects
    ./model/<conf.model>.py to be the pytorch model file.

    The function copies the model file into the cache BEFORE loading, for easy reproducibility.
    """

    src_path = os.path.join('.', 'models', conf.model + '.py')
    dst_path = os.path.join(cache_folder, conf.model + '.py')

    # (re-) copy the model file
    if os.path.exists(dst_path): os.remove(dst_path)
    shutil.copyfile(src_path, dst_path)

    # load and build
    network = absolute_import(dst_path)
    network = network.build(conf, backbone, 'train')

    # multi-gpu
    #network = torch.nn.DataParallel(network)

    # load SGD
    if conf.solver_type.lower() == 'sgd':

        mo = conf.momentum
        wd = conf.weight_decay
        lr = adjust_lr(conf)

        optimizer = fluid.optimizer.MomentumOptimizer(
            learning_rate=lr,
            momentum=mo,
            regularization=fluid.regularizer.L2Decay(wd),
            parameter_list=network.parameters())

    # load adam
    elif conf.solver_type.lower() == 'adam':

        lr = conf.lr
        wd = conf.weight_decay
        optimizer = fluid.optimizer.Adam(
            learning_rate=lr,
            regularization=fluid.regularizer.L2Decay(wd),
            parameter_list=network.parameters())

    # load adamax
    elif conf.solver_type.lower() == 'adamax':

        lr = conf.lr
        wd = conf.weight_decay
        optimizer = fluid.optimizer.Adamax(
            learning_rate=lr,
            regularization=fluid.regularizer.L2Decay(wd),
            parameter_list=network.parameters())

    return network, optimizer  #, lr


def intersect(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of intersect between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the intersect in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        # np.ndarray
        if data_type == np.ndarray:
            max_xy = np.minimum(
                box_a[:, 2:4], np.expand_dims(
                    box_b[:, 2:4], axis=1))
            min_xy = np.maximum(
                box_a[:, 0:2], np.expand_dims(
                    box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == fluid.core_avx.VarBase:
            max_xy = fluid.layers.elementwise_min(box_a[:, 2:], box_b[:, 2:])
            min_xy = fluid.layers.elementwise_max(box_a[:, :2], box_b[:, :2])
            inter = fluid.layers.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.maximum(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))


def iou(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
        union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) - inter

        # np.ndarray
        if data_type == np.ndarray:
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))


def iou_ign(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of overap of box_b has within box_a, which is handy for dealing with ignore regions.
    Hence, assume that box_b are ignore regions and box_a are anchor boxes, then we may want to know how
    much overlap the anchors have inside of the ignore regions (hence ignore area_b!)

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
        union = np.expand_dims(area_a, 0) + np.expand_dims(area_b,
                                                           1) * 0 - inter * 0

        # torch and numpy have different calls for transpose
        if data_type == np.ndarray:
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

    else:
        raise ValueError('unknown mode {}'.format(mode))


def to_int(string, dest="I"):
    return struct.unpack(dest, string)[0]


def parse_shape_from_file(filename):
    with open(filename, "rb") as file:
        version = file.read(4)
        lod_level = to_int(file.read(8), dest="Q")
        for i in range(lod_level):
            _size = to_int(file.read(8), dest="Q")
            _ = file.read(_size)
        version = file.read(4)
        tensor_desc_size = to_int(file.read(4))
        tensor_desc = VarType.TensorDesc()
        tensor_desc.ParseFromString(file.read(tensor_desc_size))
    return tuple(tensor_desc.dims)


def load_vars(train_prog, path):
    """
    loads a paddle models vars from a given path.
    """
    load_vars = []
    load_fail_vars = []

    def var_shape_matched(var, shape):
        var_exist = os.path.exists(os.path.join(path, var.name))
        if var_exist:
            var_shape = parse_shape_from_file(os.path.join(path, var.name))
            return var_shape == shape
        return False

    for x in train_prog.list_vars():
        if isinstance(x, fluid.framework.Parameter):
            shape = tuple(fluid.global_scope().find_var(x.name).get_tensor()
                          .shape())
            if var_shape_matched(x, shape):
                load_vars.append(x)
            else:
                load_fail_vars.append(x)

    return load_vars, load_fail_vars


def log_stats(tracker, iteration, start_time, start_iter, max_iter, skip=1):
    """
    This function writes the given stats to the log / prints to the screen.
    Also, computes the estimated time arrival (eta) for completion and (dt) delta time per iteration.

    Args:
        tracker (array): dictionary array tracker objects. See below.
        iteration (int): the current iteration
        start_time (float): starting time of whole experiment
        start_iter (int): starting iteration of whole experiment
        max_iter (int): maximum iteration to go to

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    display_str = 'iter: {}'.format((int((iteration) / skip)))

    # compute eta
    time_str, dt = compute_eta(start_time, iteration - start_iter,
                               max_iter - start_iter)

    # cycle through all tracks
    last_group = ''
    for key in sorted(tracker.keys()):

        if type(tracker[key]) == list:

            # compute mean
            meanval = np.mean(tracker[key])

            # get properties
            format = tracker[key + '_obj'].format
            group = tracker[key + '_obj'].group
            name = tracker[key + '_obj'].name

            # logic to have the string formatted nicely
            # basically roughly this format:
            #   iter: {}, group_1 (name: val, name: val), group_2 (name: val), dt: val, eta: val
            if last_group != group and last_group == '':
                display_str += (', {} ({}: ' + format).format(group, name,
                                                              meanval)

            elif last_group != group:
                display_str += ('), {} ({}: ' + format).format(group, name,
                                                               meanval)

            else:
                display_str += (', {}: ' + format).format(name, meanval)

            last_group = group

    # append dt and eta
    display_str += '), dt: {:0.2f}, eta: {}'.format(dt, time_str)

    # log
    logging.info(display_str)


def display_stats(vis,
                  tracker,
                  iteration,
                  start_time,
                  start_iter,
                  max_iter,
                  conf_name,
                  conf_pretty,
                  skip=1):
    """
    This function plots the statistics using visdom package, similar to the log_stats function.
    Also, computes the estimated time arrival (eta) for completion and (dt) delta time per iteration.

    Args:
        vis (visdom): the main visdom session object
        tracker (array): dictionary array tracker objects. See below.
        iteration (int): the current iteration
        start_time (float): starting time of whole experiment
        start_iter (int): starting iteration of whole experiment
        max_iter (int): maximum iteration to go to
        conf_name (str): experiment name used for visdom display
        conf_pretty (str): pretty string with ALL configuration params to display

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    # compute eta
    time_str, dt = compute_eta(start_time, iteration - start_iter,
                               max_iter - start_iter)

    # general info
    info = 'Experiment: <b>{}</b>, Eta: <b>{}</b>, Time/it: {:0.2f}s\n'.format(
        conf_name, time_str, dt)
    info += conf_pretty

    # replace all newlines and spaces with line break <br> and non-breaking spaces &nbsp
    info = info.replace('\n', '<br>')
    info = info.replace(' ', '&nbsp')

    # pre-formatted html tag
    info = '<pre>' + info + '</pre'

    # update the info window
    vis.text(
        info, win='info', opts={'title': 'info',
                                'width': 500,
                                'height': 350})

    # draw graphs for each track
    for key in sorted(tracker.keys()):

        if type(tracker[key]) == list:
            meanval = np.mean(tracker[key])
            group = tracker[key + '_obj'].group
            name = tracker[key + '_obj'].name

            # new data point
            vis.line(
                X=np.array([(iteration + 1)]),
                Y=np.array([meanval]),
                win=group,
                name=name,
                update='append',
                opts={
                    'showlegend': True,
                    'title': group,
                    'width': 500,
                    'height': 350,
                    'xlabel': 'iteration'
                })


def compute_stats(tracker, stats):
    """
    Copies any arbitary statistics which appear in 'stats' into 'tracker'.
    Also, for each new object to track we will secretly store the objects information
    into 'tracker' with the key as (group + name + '_obj'). This way we can retrieve these properties later.

    Args:
        tracker (array): dictionary array tracker objects. See below.
        stats (array): dictionary array tracker objects. See below.

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    # through all stats
    for stat in stats:

        # get properties
        name = stat['name']
        group = stat['group']
        val = stat['val']

        # convention for identificaiton
        id = group + name

        # init if not exist?
        if not (id in tracker): tracker[id] = []

        # # convert tensor to numpy
        # if type(val) == torch.Tensor:
        #     val = val.cpu().detach().numpy()

        # store
        tracker[id].append(val)

        # store object info
        obj_id = id + '_obj'
        if not (obj_id in tracker):
            stat.pop('val', None)
            tracker[id + '_obj'] = stat


def init_training_paths(conf_name, use_tmp_folder=None):
    """
    Simple function to store and create the relevant paths for the project,
    based on the base = current_working_dir (cwd). For this reason, we expect
    that the experiments are run from the root folder.

    data    =  ./data
    output  =  ./output/<conf_name>
    weights =  ./output/<conf_name>/weights
    results =  ./output/<conf_name>/results
    logs    =  ./output/<conf_name>/log

    Args:
        conf_name (str): configuration experiment name (used for storage into ./output/<conf_name>)
    """

    # make paths
    paths = edict()
    paths.base = os.getcwd()
    paths.data = os.path.join(paths.base, 'dataset')
    paths.output = os.path.join(os.getcwd(), 'output', conf_name)
    paths.weights = os.path.join(paths.output, 'weights')
    paths.logs = os.path.join(paths.output, 'log')

    if use_tmp_folder:
        paths.results = os.path.join(paths.base, '.tmp_results', conf_name,
                                     'results')
    else:
        paths.results = os.path.join(paths.output, 'results')

    # make directories
    mkdir_if_missing(paths.output)
    mkdir_if_missing(paths.logs)
    mkdir_if_missing(paths.weights)
    mkdir_if_missing(paths.results)

    return paths
