import paddle
import paddle.fluid as fluid


class LogisticModel(object):
    """Logistic model with L2 regularization."""

    def build_model(self,
                    model_input,
                    vocab_size,
                    l2_penalty=None,
                    **unused_params):
        """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
        logit = fluid.layers.fc(
            input=model_input,
            size=vocab_size,
            act=None,
            name='logits_clf',
            param_attr=fluid.ParamAttr(
                name='logistic.weights',
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=fluid.ParamAttr(
                name='logistic.bias',
                initializer=fluid.initializer.MSRA(uniform=False)))
        output = fluid.layers.sigmoid(logit)
        return output, logit
