import paddle
import paddle.fluid as fluid


class LogisticModel(object):
    """Logistic model with L2 regularization."""

    def create_model(self,
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
        logits = fluid.layers.fc(
            input=model_input,
            size=vocab_size,
            act=None,
            name='logits_clf',
            param_attr=fluid.ParamAttr(
                name='logits_clf_weights',
                initializer=fluid.initializer.MSRA(uniform=False),
                regularizer=fluid.regularizer.L2DecayRegularizer(l2_penalty)),
            bias_attr=fluid.ParamAttr(
                name='logits_clf_bias',
                regularizer=fluid.regularizer.L2DecayRegularizer(l2_penalty)))
        output = fluid.layers.sigmoid(logits)
        return {'predictions': output, 'logits': logits}
