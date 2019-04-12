#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

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
