"""
"""

from .register import get_registered_layers
#custom layer import begins

import axpy
import flatten
import argmax
import reshape
import roipooling
import priorbox
import permute
import detection_out
import normalize
import select
import crop
import power
import reduction

#custom layer import ends

custom_layers = get_registered_layers()


def set_args(f, params, node=None):
    """ set args for function 'f' using the parameters in node.layer.parameters

    Args:
        f (function): a python function object
        params (object): a object contains attributes needed by f's arguments

    Returns:
        arg_names (list): a list of argument names
        kwargs (dict): a dict contains needed arguments
    """
    from ..protobuf_to_dict import protobuf_to_dict

    argc = f.__code__.co_argcount
    arg_list = f.__code__.co_varnames[0:argc]

    kwargs = {}
    for arg_name in arg_list:
        if arg_name in params:
            kwargs[arg_name] = params[arg_name]

    if node is not None and len(node.metadata):
        kwargs.update(node.metadata)

    return arg_list, kwargs


def has_layer(kind):
    """ test whether this layer exists in custom layer
    """
    return kind in custom_layers


def compute_output_shape(kind, node):
    assert kind in custom_layers, "layer[%s] not exist in custom layers" % (
        kind)
    shape_func = custom_layers[kind]['shape']

    parents = node.parents
    inputs = [list(p.output_shape) for p in parents]
    arg_names, kwargs = set_args(shape_func, node.params)

    if len(inputs) == 1:
        inputs = inputs[0]

    return shape_func(inputs, **kwargs)


def make_node(template, kind, node):
    """ make a PaddleNode for custom layer which means construct
        a piece of code to define a layer implemented in 'custom_layers'

    Args:
        @template (PaddleNode): a factory to new a instance of PaddleNode
        @kind (str): type of custom layer
        @node (graph.Node): a layer in the net

    Returns:
        instance of PaddleNode
    """
    assert kind in custom_layers, "layer[%s] not exist in custom layers" % (
        kind)

    layer_func = custom_layers[kind]['layer']

    #construct arguments needed by custom layer function from node's parameters
    arg_names, kwargs = set_args(layer_func, node.params, node)

    return template('custom_layer', kind, **kwargs)


def make_custom_layer(kind, inputs, name, *args, **kwargs):
    """ execute a custom layer which is implemented by users

    Args:
        @kind (str): type name of this layer
        @inputs (vars): variable list created by fluid
        @namme (str): name for this layer
        @args (tuple): other positional arguments
        @kwargs (dict): other kv arguments

    Returns:
        output (var): output variable for this layer
    """
    assert kind in custom_layers, "layer[%s] not exist in custom layers" % (
        kind)

    layer_func = custom_layers[kind]['layer']
    return layer_func(inputs, name, *args, **kwargs)
