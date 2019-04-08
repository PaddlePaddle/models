"""
optimizer calss
"""

import paddle.fluid as fluid


class SGDOptimizer(object):
    """
    SGD
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        self.learning_rate = conf_dict["optimizer"]["learning_rate"]

    def ops(self, loss):
        """
        SGD optimizer operation
        """
        sgd = fluid.optimizer.SGDOptimizer(self.learning_rate)
        sgd.minimize(loss)


class AdamOptimizer(object):
    """
    Adam
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        self.learning_rate = conf_dict["optimizer"]["learning_rate"]
        self.beta1 = conf_dict["optimizer"]["beta1"]
        self.beta2 = conf_dict["optimizer"]["beta2"]
        self.epsilon = conf_dict["optimizer"]["epsilon"]

    def ops(self, loss):
        """
        Adam optimizer operation
        """
        adam = fluid.optimizer.AdamOptimizer(
            self.learning_rate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon)
        adam.minimize(loss)
