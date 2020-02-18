from functools import wraps
import importlib


def model_constructor(f):
    """ Wraps the function 'f' which returns the network. An extra field 'constructor' is added to the network returned
    by 'f'. This field contains an instance of the  'NetConstructor' class, which contains the information needed to
    re-construct the network, such as the name of the function 'f', the function arguments etc. Thus, the network can
    be easily constructed from a saved checkpoint by calling NetConstructor.get() function.
    """
    @wraps(f)
    def f_wrapper(*args, **kwds):
        net_constr = NetConstructor(f.__name__, f.__module__, args, kwds)
        output = f(*args, **kwds)
        if isinstance(output, (tuple, list)):
            # Assume first argument is the network
            output[0].constructor = net_constr
        else:
            output.constructor = net_constr
        return output
    return f_wrapper


class NetConstructor:
    """ Class to construct networks. Takes as input the function name (e.g. atom_resnet18), the name of the module
    which contains the network function (e.g. ltr.models.bbreg.atom) and the arguments for the network
    function. The class object can then be stored along with the network weights to re-construct the network."""
    def __init__(self, fun_name, fun_module, args, kwds):
        """
        args:
            fun_name - The function which returns the network
            fun_module - the module which contains the network function
            args - arguments which are passed to the network function
            kwds - arguments which are passed to the network function
        """
        self.fun_name = fun_name
        self.fun_module = fun_module
        self.args = args
        self.kwds = kwds

    def get(self):
        """ Rebuild the network by calling the network function with the correct arguments. """
        net_module = importlib.import_module(self.fun_module)
        net_fun = getattr(net_module, self.fun_name)
        return net_fun(*self.args, **self.kwds)
