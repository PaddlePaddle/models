Running sample code in this directory requires PaddelPaddle v0.10.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html) and make an update.

---

# Scheduled Sampling

## Overview

The goal of a sequence generation task is to maximize the probability of the target sequence given the input of the source. When training, the model uses real elements in the target sequence as the input to each step of the decoder and then maximizes the probability of the next element. The element decoded in the previous step is used as the current input to generate the next element. We can see that the probability distribution of the decoder input data in the training phase and the generation phase is not consistent in this case.

Scheduled sampling\[[1](#references)\] is a solution to the inconsistency in the distribution of input data during training and generation phases. In the early stage of training, this method uses the real elements in the target sequence as the decoder input and quickly guides the model from a randomly initialized state to a reasonable state. As training progresses, the method will gradually increase the use  of the generated elements as decoder input to solve the problem of inconsistent data distribution.

In a standard sequence-to-sequence model, if an incorrect element is generated at the beginning of the sequence, the subsequent input state will be affected, and the error will continue to accumulate as the generation process continues. Scheduled sampling uses the generated elements as the decoder input with a certain probability, so even if the previous generation steps have errors, the training process' target is still to maximize the probability of the real target sequence, so the model is still trained in the right direction. Therefore, this approach increases the fault tolerance of the model.

## Introduction to the Algorithm
Scheduled sampling is used only in the training phase of the sequence-to-sequence model and not in the generation phase.

The standard sequence-to-sequence model uses the true element, $y_{t-1}$, at the previous moment as input for the decoder to maximize the probability of the $t$-th element. Let $g_{t-1}$ be the element generated at the latest moment. The scheduled sampling algorithm will use $g_{t-1}$ as the decoder input with a certain probability.

Suppose that the $i$-th mini-batch has been trained. To control the decoder's input, the scheduled sampling algorithm defines a probability variable $\epsilon_i$ that decays as $i$ increases. Some common definitions are:

Linear attenuation: $\epsilon_i=max(\epsilon,k-c*i)$, where $\epsilon$ limits the minimum value of $\epsilon_i$, and $k$ and $c$ control the magnitude of linear attenuation.

Exponential decay: $\epsilon_i=k^i$, where $0 <k <1$, $k$ controls the magnitude of the exponential decay.

Inverse Sigmoid decay: $\epsilon_i=k/(k+exp(i/k))$, where $k>1$, $k$ also controls the magnitude of attenuation.

<p align="center">
<img src="images/decay.jpg" width="50%" align="center"><br>
Figure 1. Attenuation curves for linear attenuation, exponential decay, and inverse Sigmoid decay
</p>

As shown in Fig. 2, at time $t$ of the decoder, the scheduled sampling algorithm uses the true element $yt−1$ of the previous moment as the decoder input with probability $\epsilon_i$, and uses $g_{t-1}$ generated at the previous moment as the decoder input with probability $1-\epsilon_i$. From Figure 1, we see that as $i$ increases, $\epsilon_i$ decreases. Decoder will continue to use the generated elements as input. The data distribution during the training phase and the generation phase will gradually become more consistent.

<p align="center">
<img src="images/Scheduled_Sampling.jpg" width="50%" align="center"><br>
Figure 2. Scheduled sampling algorithm selects different elements as decoder input
</p>

## Model Implementation

Since the scheduled sampling algorithm is just an improvement over the sequence-to-sequence model, its overall implementation framework is similar to that of the sequence-to-sequence model. Thus, only the parts related to scheduled sampling are described here. For the complete code, see `network_conf.py`.

First, we import the required package and define the class `RandomScheduleGenerator` that controls the decay probability as follows:

```python
import numpy as np
import math


class RandomScheduleGenerator:
"""
The random sampling rate for scheduled sampling algoithm, which uses devcayed
sampling rate.

"""
...
```

We will now define the three methods of class `RandomScheduleGenerator`: `__init__`, `getScheduleRate`, and `processBatch`.

The `__init__` method initializes the class. The `schedule_type` parameter specifies which decay mode to use. The options are `constant`, `linear`, `exponential`, and `inverse_sigmoid`. Mode `constant` uses a fixed $\epsilon_i$ for all mini-batch; mode `linear` refers to linear attenuation; mode `exponential` refers to exponential decay; mode `inverse_sigmoid` refers to inverse sigmoid decay. Parameters `a` and `b` of the `__init__` method represent the parameters of the attenuation method, and they need to be tuned on the validation set. `self.schedule_computers` maps the decay mode to a function that calculates εi. The last line assigns the selected attenuation function to `self.schedule_computer` according to `schedule_type`.

```python
def __init__(self, schedule_type, a, b):
"""
schduled_type: is the type of the decay. It supports constant, linear,
exponential, and inverse_sigmoid right now.
a: parameter of the decay (MUST BE DOUBLE)
b: parameter of the decay (MUST BE DOUBLE)
"""
self.schedule_type = schedule_type
self.a = a
self.b = b
self.data_processed_ = 0
self.schedule_computers = {
"constant": lambda a, b, d: a,
"linear": lambda a, b, d: max(a, 1 - d / b),
"exponential": lambda a, b, d: pow(a, d / b),
"inverse_sigmoid": lambda a, b, d: b / (b + math.exp(d * a / b)),
}
assert (self.schedule_type in self.schedule_computers)
self.schedule_computer = self.schedule_computers[self.schedule_type]
```

`getScheduleRate` calculates $\epsilon_i$ based on the decay function and the amount of data already processed.

```python
def getScheduleRate(self):
"""
Get the schedule sampling rate. Usually not needed to be called by the users
"""
return self.schedule_computer(self.a, self.b, self.data_processed_)

```

The `processBatch` method is sampled according to the probability value $\epsilon_i$, and it output `indexes`. Each element in `indexes` has ϵi probability to be assigned `0`, $1-\epsilon_i$ to be assigned `1`. `indexes` determines whether the decoder's input is a real element or a generated element. A value of `0` indicates that the real element is used, and a value of `1` indicates that the generated element is used.

```python
def processBatch(self, batch_size):
"""
Get a batch_size of sampled indexes. These indexes can be passed to a
MultiplexLayer to select from the grouth truth and generated samples
from the last time step.
"""
rate = self.getScheduleRate()
numbers = np.random.rand(batch_size)
indexes = (numbers >= rate).astype('int32').tolist()
self.data_processed_ += batch_size
return indexes
```

Scheduled sampling algorithm needs to add to the sequence-to-sequence model another input variable, `true_token_flag`, to control the decoder input.

```python
true_token_flags = paddle.layer.data(
name='true_token_flag',
type=paddle.data_type.integer_value_sequence(2))
```

Here, we also need to encapsulate the original reader and add a data generator to `true_token_flag`. We use linear decay as an example to show how to call `RandomScheduleGenerator` defined above to generate input data for `true_token_flag`.

```python
def gen_schedule_data(reader,
schedule_type="linear",
decay_a=0.75,
decay_b=1000000):
"""
Creates a data reader for scheduled sampling.

Output from the iterator that created by original reader will be
appended with "true_token_flag" to indicate whether to use true token.

:param reader: the original reader.
:type reader: callable
:param schedule_type: the type of sampling rate decay.
:type schedule_type: str
:param decay_a: the decay parameter a.
:type decay_a: float
:param decay_b: the decay parameter b.
:type decay_b: float

:return: the new reader with the field "true_token_flag".
:rtype: callable
"""
schedule_generator = RandomScheduleGenerator(schedule_type, decay_a, decay_b)

def data_reader():
for src_ids, trg_ids, trg_ids_next in reader():
yield src_ids, trg_ids, trg_ids_next, \
[0] + schedule_generator.processBatch(len(trg_ids) - 1)

return data_reader
```

This code appends the input data that controls the decoder input after the original input data (ie, source sequence element `src_ids`, target sequence element `trg_ids`, and an element in the target sequence `trg_ids_next`). Since the first element of the decoder is the sequence starter, we set the first element of the appended data to `0`, indicating that the first operation of the decoder always uses the first element of the real target sequence (ie, the sequence starter).

The decoder function called by each step of `recurrent_group` during training is as follows:

```python
def gru_decoder_with_attention_train(enc_vec, enc_proj, true_word,
true_token_flag):
"""
The decoder step for training.
:param enc_vec: the encoder vector for attention
:type enc_vec: LayerOutput
:param enc_proj: the encoder projection for attention
:type enc_proj: LayerOutput
:param true_word: the ground-truth target word
:type true_word: LayerOutput
:param true_token_flag: the flag of using the ground-truth target word
:type true_token_flag: LayerOutput
:return: the softmax output layer
:rtype: LayerOutput
"""

decoder_mem = paddle.layer.memory(
name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)

context = paddle.networks.simple_attention(
encoded_sequence=enc_vec,
encoded_proj=enc_proj,
decoder_state=decoder_mem)

gru_out_memory = paddle.layer.memory(
name='gru_out', size=target_dict_dim)

generated_word = paddle.layer.max_id(input=gru_out_memory)

generated_word_emb = paddle.layer.embedding(
input=generated_word,
size=word_vector_dim,
param_attr=paddle.attr.ParamAttr(name='_target_language_embedding'))

current_word = paddle.layer.multiplex(
input=[true_token_flag, true_word, generated_word_emb])

decoder_inputs = paddle.layer.fc(
input=[context, current_word],
size=decoder_size * 3,
act=paddle.activation.Linear(),
bias_attr=False)

gru_step = paddle.layer.gru_step(
name='gru_decoder',
input=decoder_inputs,
output_mem=decoder_mem,
size=decoder_size)

out = paddle.layer.fc(
name='gru_out',
input=gru_step,
size=target_dict_dim,
act=paddle.activation.Softmax())
return out
```

The function uses the `memory` layer `gru_out_memory` to memorize the elements generated at the last moment, and select the word with the highest probability as the generated word. The `multiplex` layer makes a choice between the true element, `true_word`, and the generated element, `generated_word`, and uses the result as the decoder input. The `multiplex` layer uses three inputs, `true_token_flag`, `true_word`, and `generated_word_emb`. For each of these three inputs, if `true_token_flag` is 0, then the `multiplex` layer outputs the corresponding element in `true_word`; if `true_token_flag` is 1, then the `multiplex` layer outputs the corresponding element in `generated_word_emb`.

## References

[1] Bengio S, Vinyals O, Jaitly N, et al. [Scheduled sampling for sequence prediction with recurrent neural networks](http://papers.nips.cc/paper/5956-scheduled-sampling-for-sequence-prediction-with-recurrent-neural-networks)//Advances in Neural Information Processing Systems. 2015: 1171-1179.
