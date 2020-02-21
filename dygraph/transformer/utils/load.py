import pickle
import six
import warnings
from functools import partial

import paddle.fluid as fluid

def load_dygraph(model_path, keep_name_table=False):
    """
    To load python2 saved models in python3.
    """
    try:
        para_dict, opti_dict = fluid.load_dygraph(model_path, keep_name_table)
        return para_dict, opti_dict
    except UnicodeDecodeError:
        warnings.warn(
            "An UnicodeDecodeError is catched, which might be caused by loading "
            "a python2 saved model. Encoding of pickle.load would be set and "
            "load again automatically.")
        if six.PY3:
            load_bak = pickle.load
            pickle.load = partial(load_bak, encoding="latin1")
            para_dict, opti_dict = fluid.load_dygraph(model_path, keep_name_table)
            pickle.load = load_bak
            return para_dict, opti_dict
