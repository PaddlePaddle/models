# Scheduled Sampling

## 概述
序列生成任务的训练目标是在给定源输入的条件下，最大化目标序列的概率。训练时该模型将目标序列中的真实元素作为解码阶段每一步的输入，然后最大化下一个元素的概率。生成时上一步解码得到的元素被用作当前的输入，然后生成下一个元素。可见这种情况下训练阶段和生成阶段的解码层输入数据的概率分布并不一致。如果序列前面生成了错误的元素，后面的输入状态将会收到影响，而该误差会随着生成过程不断向后累积。
Scheduled Sampling是一种解决训练和生成时输入数据分布不一致的方法。在训练早期该方法主要使用真实元素作为解码输入，以将模型从随机初始化的状态快速引导至一个合理的状态。随着训练的进行该方法会逐渐更多的使用生成元素作为解码输入，以解决数据分布不一致的问题。

## 算法简介
Scheduled Sampling主要应用在Sequence to Sequence模型的训练上，而生成阶段则不需要使用。
解码阶段在生成第`t`个元素时，标准Sequence to Sequence模型使用上一时刻的真实元素`y(t-1)`作为输入。设上一时刻生成的元素为`g(t-1)`，Scheduled Sampling算法会以一定概率使用`g(t-1)`作为解码输入。
设当前已经训练到了第`i`个mini-batch，在`t`时刻Scheduled Sampling以概率`epsilon_i`使用上一时刻的真实元素`y(t-1)`作为解码输入，以概率`1-epsilon_i`使用上一时刻生成的元素`g(t-1)`作为解码输入。
随着`i`的增大`epsilon_i`会不断减小，解码阶段将不断倾向于使用生成的元素作为输入，训练阶段和生成阶段的数据分布将变得越来越一致。
`epsilon_i`可以使用不同的方式衰减，常见的方式有：

 - 线性衰减：`epsilon_i=max(epsilon,k-c*i)`，其中`epsilon`限制`epsilon_i`的最小值，`k`和`c`控制线性衰减的幅度。
 - 指数衰减：`epsilon_i=k^i`，其中`0<k<1`，`k`控制着指数衰减的幅度。
 - 反向Sigmoid衰减：`epsilon_i=k/(k+exp(i/k))`，其中`k>1`，`k`同样控制衰减的幅度。

## 模型实现
由于Scheduled Sampling是对Sequence to Sequence模型的改进，其整体实现框架与Sequence to Sequence模型较为相似。为突出本文重点，这里仅介绍与Scheduled Sampling相关的部分，完整的代码见`scheduled_sampling.py`。

首先定义控制衰减概率的类`RandomScheduleGenerator`，如下：
```python
import numpy as np
import math


class RandomScheduleGenerator:
    """
    The random sampling rate for scheduled sampling algoithm, which uses devcayed
    sampling rate.
    """

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

    def getScheduleRate(self):
        """
        Get the schedule sampling rate. Usually not needed to be called by the users
        """
        return self.schedule_computer(self.a, self.b, self.data_processed_)

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
其中`__init__`方法定义了几种不同的衰减概率，`processBatch`方法根据该概率进行采样，最终确定解码时是使用真实元素还是使用生成的元素。


这里对数据reader进行封装，加入从`RandomScheduleGenerator`采样得到的`true_token_flag`作为另一组数据输入，控制解码使用的元素。

```python
schedule_generator = RandomScheduleGenerator("linear", 0.75, 1000000)

def gen_schedule_data(reader):
    """
    Creates a data reader for scheduled sampling.

    Output from the iterator that created by original reader will be
    appended with "true_token_flag" to indicate whether to use true token.

    :param reader: the original reader.
    :type reader: callable

    :return: the new reader with the field "true_token_flag".
    :rtype: callable
    """

    def data_reader():
        for src_ids, trg_ids, trg_ids_next in reader():
            yield src_ids, trg_ids, trg_ids_next, \
                  [0] + schedule_generator.processBatch(len(trg_ids) - 1)

    return data_reader
```

训练时`recurrent_group`每一步调用的解码函数如下：

```python
    def gru_decoder_with_attention_train(enc_vec, enc_proj, true_word,
                                         true_token_flag):
        """
        The decoder step for training.
        :param enc_vec: the encoder vector for attention
        :type enc_vec: Layer
        :param enc_proj: the encoder projection for attention
        :type enc_proj: Layer
        :param true_word: the ground-truth target word
        :type true_word: Layer
        :param true_token_flag: the flag of using the ground-truth target word
        :type true_token_flag: Layer
        :return: the softmax output layer
        :rtype: Layer
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

        with paddle.layer.mixed(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += paddle.layer.full_matrix_projection(input=context)
            decoder_inputs += paddle.layer.full_matrix_projection(
                input=current_word)

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        with paddle.layer.mixed(
                name='gru_out',
                size=target_dict_dim,
                bias_attr=True,
                act=paddle.activation.Softmax()) as out:
            out += paddle.layer.full_matrix_projection(input=gru_step)

        return out
```

该函数使用`memory`层`gru_out_memory`记忆不同时刻生成的元素，并使用`multiplex`层选择是否使用生成的元素作为解码输入。

### 训练结果待调参完成后补充
