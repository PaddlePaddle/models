""" this module provides 'register' for registering customized layers
"""

g_custom_layers = {}


def register(kind, shape, layer):
    """ register a custom layer or a list of custom layers

    Args:
        @kind (str or list): type name of the layer
        @shape (function): a function to generate the shape of layer's output
        @layer (function): a function to generate the shape of layer's output

    Returns:
        None
    """
    assert type(shape).__name__ == 'function', 'shape should be a function'
    assert type(layer).__name__ == 'function', 'layer should be a function'

    if type(kind) is str:
        kind = [kind]
    else:
        assert type(
            kind) is list, 'invalid param "kind" for register, not a list or str'

    for k in kind:
        assert type(
            k) is str, 'invalid param "kind" for register, not a list of str'
        assert k not in g_custom_layers, 'this type[%s] has already been registered' % (
            k)
        print('register layer[%s]' % (k))
        g_custom_layers[k] = {'shape': shape, 'layer': layer}


def get_registered_layers():
    return g_custom_layers
