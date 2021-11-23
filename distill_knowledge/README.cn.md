运行本目录下的程序示例使用PaddlePaddle v0.12.0 版本。如果您的PaddlePaddle安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明更新PaddlePaddle安装版本。

# 神经网络知识蒸馏
本文介绍了如何使用PaddlePaddle实现文章《Distilling the Knowledge in a Neural Network》[^distill](2014 NIPS workshop)在MNIST[^mnist]数据上训练神经网络的方法与功能。

## 方法概述
知识蒸馏方法的目的是从比较大的复杂的大模型中提取信息，用于较小模型的训练，从而可以使得小模型达到更好的效果。以MNIST分类任务为例，大模型通常可以产生具有高置信度的分类结果，除了我们需要的结果之外，还有更多的信息隐藏在输出的软目标(soft targets)概率的比例中，我们可以利用大模型输出的软目标概率来作为小模型训练的监督信息，结合正常的训练目标从而提升小模型的性能。在该方法中，我们通过提升最终softmax层的temperature来获得更加合适的软目标概率。

## 基本训练流程
1.训练大模型: 使用hard targets, 也就是正常的label训练大模型。
2.计算训练样本的soft targets: 利用训练好的大模型，在softmax函数中不同的temperature下产生训练样本的soft targets，公式如下：
$$ q_i = \frac{exp(z_i / T)}{\sum_j exp(z_j / T)} $$
其中，$q_i$ 是产生的soft targets， $z_i$ 是network output， $T$ 是 temperature。
3.利用soft targets与 hard targets训练小模型: 小模型使用相同的temperature与soft targets进行crossentropy就算soft损失，使用hard targets计算hard 损失，由于soft targets被缩放了 $\frac{1}{T^2}$, 为了平衡训练的梯度，需要将soft 损失扩大 $T^2$.
4.在训练结束后，使用整成的推断方式测试小网络性能，并且与正常训练方式进行比较。

更多细节讨论参考论文[^distill]

## 代码结构

| File&Folder      |     Description |  
| :-------- | :-------- |
| train.py    |   Training script |  
| infer.py    |   Predcition using the trained model script |  
| utils.py    |   Dataset loading classes and functions |
| mnist_prepare.py    |   Preparing Mnist Datasets |
| ./data/     | The folder stores dataset and soft targets dataset |
| ./models/   |   The folder stores trained models |  
| ./images/   |  Illustration graphs  |  


## 数据准备
运行`mnist_prepare.py` 使用PaddlePaddle dataset 函数准备原始的MNIST数据 。MNIST数据存储在文件 `./data/mnist.npz` ， 所有的数据都被标准化到[-1, 1].
```python
reader = paddle.dataset.mnist.train()
~
np.savez('./data/mnist.npz', train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
```

## Training
**首先**, 我们训练一个单个大的神经网络，我们使用一个两层，每层1200隐藏节点的全连接网络作为大网络。
```python
def mlp_teacher(img, drop_prob):
    h1 = fluid.layers.fc(input=img, size=1200, act='relu')
    drop1 = fluid.layers.dropout(h1, dropout_prob=drop_prob)
    h2 = fluid.layers.fc(input=drop1, size=1200, act='relu')
    drop2 = fluid.layers.dropout(h2, dropout_prob=drop_prob)
    logits = fluid.layers.fc(input=drop2, size=10, act=None)
    return logits
```
运行`python train.py --phase teacher --drop_prob 0.4`  来训练大的神经网络，训练好的模型参数存储在 `.\models\teacher_net`.

**然后**，我们运行  `python infer.py --phase teacher --temp 4.0` 来产生训练数据对应的soft targets。 我们可以设置不同的 `temp` 超参数来产生不同的soft targets，不同的`temp`参数产生的soft targets对小模型的影响是不同的。为了保持新产生的soft targets与训练样本一致，训练小网络的数据存储在`./data/mnist_soft_{temp}.npz`.
```python
        print("Generating soft-targets.......")
        g_iternum = train_set.num_examples // G_batch_size
        soft_list = []
        for i in range(g_iternum):
            g_batch = list(zip(train_x[i*G_batch_size:(i+1)*G_batch_size], train_y[i*G_batch_size:(i+1)*G_batch_size]))
            soft_targets, gb_acc = exe.run(inference_program, feed=feeder.feed(g_batch),
                    fetch_list=[temp_softmax_logits, batch_acc])
            soft_list.append(soft_targets)

        train_y_soft = np.vstack(soft_list)
        print("saving soft targets")
        np.savez(data_dir+'mnist_soft_{}.npz'.format(temp),
                 train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, train_y_soft=train_y_soft)
```

**然后**, 运行 `python train.py --phase student --dropout 0.1 --stu_hsize 30 --temp 4.0 --use_soft True`  结合soft targets训练小网络。训练小网络时，temperature应该与大的网络产生的soft targets的temperature一致。 Softmax with temperature 函数定义如下:
```python
def softmax_with_temperature(logits, temp=1.0):
    logits_with_temp = logits/temp
    _softmax = fluid.layers.softmax(logits_with_temp)
    return _softmax
```
定义目标函数如下：
```python
def soft_crossentropy(input, label):
    epsilon = 1e-8
    eps = fluid.layers.ones(shape=[1], dtype='float32') * epsilon
    loss = reduce_sum(-1.0 * label * log(elementwise_max(input, eps)), dim=1, keep_dim=True)
    return loss
~~
softmax_logits = fluid.layers.softmax(logits)
temp_softmax_logits = softmax_with_temperature(logits, temp=temp)
hard_loss = soft_crossentropy(input=softmax_logits, label=label)
soft_loss = soft_crossentropy(input=temp_softmax_logits, label=soft_label)
```
小网络也是一个两层全连接网络，但是每层的隐含单元数量要少很多。
```python
def mlp_student(img, drop_prob, h_size):
    h1 = fluid.layers.fc(input=img, size=h_size, act='relu')
    drop1 = fluid.layers.dropout(h1, dropout_prob=drop_prob)
    h2 = fluid.layers.fc(input=drop1, size=h_size, act='relu')
    drop2 = fluid.layers.dropout(h2, dropout_prob=drop_prob)
    logits = fluid.layers.fc(input=drop2, size=10, act=None)
    return logits
```

**最后**，为了验证使用soft targets的有效性，我们需要要比较小网络在使用和不适用soft targets时候测试集性能的表现。 运行 `python train.py --phase student --stu_hsize 30 --drop_prob 0.1 `  训练小网络不使用 soft targets，运行`python infer.py  --phase student --stu_hsize 30` 获得测试性能。运行 `python infer.py --phase student --stu_hsize 30 --use_soft True`. 获得使用soft targets训练的小网络的测试性能。

## Results
我们使用SGD-Momentum优化器，0.001的学习率训练大的网络200 个周期， dropout 比率设置为0.4。训练两层1200隐含单元的全连接的大网络， 最终测试结果为99.33%的正确率。为了展示知识蒸馏方法的有效性，我们用两层30隐含单元的全连接网络作为小网络。如果使用SGD 优化器，在0.001学习率下，正常训练小网络，得到的测试结果为94.38%。而结合temperature 为 4.0的soft target，在同样条件下训练的小网络，测试集结果为97.01%。可以看到，使用soft targets有了非常明显的提高。
| Methods      |  Test Accuracy |  
| :-------- | --------:|
| TeacherNet    |  99.33% |
| StudentNet 30units without soft targets | 94.38%  |  
| StudentNet 30units with 4.0temp soft targets | 97.01%  |

训练过程中的测试集准确率如下图所示，可以看到使用soft targets训练的小网络的收敛速度快于不使用soft targets的网络。
![收敛](https://github.com/likesiwell/models/blob/distill-branch/distill_knowledge/images/plots.png)


## 参考文献
[^distill]: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

[^mnist]: [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
