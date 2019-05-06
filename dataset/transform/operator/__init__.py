""" operators for processing samples like decode/resize/crop images
"""

from . import base

def build(ops):
    """ build a mapper for operator config in 'ops'

    Args:
        @ops (list of dict): configs for oprators, eg:
            [{'name': 'DecodeImage', 'params': {'to_rgb': True}}, {xxx}]

    Returns:
        a mapper function which accept one argument 'sample' and
        return the processed result
    """
    mappers = []
    op_repr = []
    for op in ops:
        op_func = getattr(base, op['name'])
        params = None if 'params' not in op else op['params']
        o = op_func(params)
        mappers.append(o)
        op_repr.append('{%s}' % str(o))
    op_repr = '[%s]' % ','.join(op_repr)

    def _mapper(sample):
        context = {}
        for f in mappers:
            out = f(sample, context)
            sample = out
        return out

    _mapper.ops = op_repr
    return _mapper

