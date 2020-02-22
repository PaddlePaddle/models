from pytracking.libs import TensorDict


class BaseActor:
    """ Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss"""

    def __init__(self, net, objective):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective

    def train(self):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train()

    def eval(self):
        """ Set network to eval mode"""
        self.net.eval()
