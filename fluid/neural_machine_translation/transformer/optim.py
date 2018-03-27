import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers


class LearningRateScheduler(object):
    """
    Wrapper for learning rate scheduling as described in the Transformer paper.
    LearningRateScheduler adapts the learning rate externally and the adapted
    learning rate will be feeded into the main_program as input data.
    """

    def __init__(self,
                 d_model,
                 warmup_steps,
                 place,
                 learning_rate=0.001,
                 current_steps=0,
                 name="learning_rate"):
        self.current_steps = current_steps
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.learning_rate = layers.create_global_var(
            name=name,
            shape=[1],
            value=float(learning_rate),
            dtype="float32",
            persistable=True)
        self.place = place

    def update_learning_rate(self, data_input):
        self.current_steps += 1
        lr_value = np.power(self.d_model, -0.5) * np.min([
            np.power(self.current_steps, -0.5),
            np.power(self.warmup_steps, -1.5) * self.current_steps
        ])
        lr_tensor = fluid.LoDTensor()
        lr_tensor.set(np.array([lr_value], dtype="float32"), self.place)
        data_input[self.learning_rate.name] = lr_tensor
