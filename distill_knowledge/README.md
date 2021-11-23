The codes in this example is tested using PaddlePaddle v0.12.0. You can install this version according to [Installation Document](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html).

# Distill the Knowledge in a Neural Network
In this example, we introduce how to use PaddlePaddle to implement the approach described in Distilling the Knowledge in a Neural Network》[^distill](2014 NIPS workshop). We demonstrate the approach on MNIST[^mnist] classification problem.
## Introduction
The aim of knowledge distillation is to extract knowledge from the large models which can facilitate the training of small models. We can take the classification problem on MNIST as an example. The trained large models can usually generate results with high confidence. In addition to the classification results we demanded, more valuable knowledge are provided in the soft predicted probabilites. We can utilize these soft targets of training data produced by the large model as additional supervised information to train small models. The experiments shows that the performance of small models improves by using appropriate temperature soft targets.

## Training Procedure
1.Train a single large model: Using hard targets to train a single large model.
2.Compute the soft targets of training examples: Using the trained large model to generate the soft targets of training examples in different temperature. The softmax function with temperature is formuated as:
$$ q_i = \frac{exp(z_i / T)}{\sum_j exp(z_j / T)}, $$
where  $q_i$ is the produced soft targets, $z_i$ is network ouputs, $T$ denotes temperature.
3.Train small a network with soft targets and hard targets: The soft targets loss is the crossentropy between soft targets and the same temperature softmax predictions. In order to balance the magnitude of the gradients between hard loss and soft loss, the soft loss is scaled by $T^2$.
4.After training, we test and compare the testing performance of small networks trained with and without soft targets.

More discussions and details, please refer to original paper[^distill].

## Directory Overview
| File&Folder      |     Description |  
| :-------- | :-------- |
| train.py    |   Training script |  
| infer.py    |   Predcition using the trained model script |  
| utils.py    |   Dataset loading classes and functions |
| mnist_prepare.py    |   Preparing Mnist Datasets |
| ./data/     | The folder stores dataset and soft targets dataset |
| ./models/   |   The folder stores trained models |  
| ./images/   |  Illustration graphs  |

## DataSet Prepare
Run `mnist_prepare.py` to prepare the original mnist data by using paddle dataset. The MNIST is stored in  `./data/mnist.npz` and the character images are standardized to range [-1, 1].
```python
reader = paddle.dataset.mnist.train()
~
np.savez('./data/mnist.npz', train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
```

## Training
**First**, we should train a single large neural network. We use a two layer fully connected network with 1200 hidden units each layer.
```python
def mlp_teacher(img, drop_prob):
    h1 = fluid.layers.fc(input=img, size=1200, act='relu')
    drop1 = fluid.layers.dropout(h1, dropout_prob=drop_prob)
    h2 = fluid.layers.fc(input=drop1, size=1200, act='relu')
    drop2 = fluid.layers.dropout(h2, dropout_prob=drop_prob)
    logits = fluid.layers.fc(input=drop2, size=10, act=None)
    return logits
```
Run `python train.py --phase teacher --drop_prob 0.4`  to train the large network. The trained parameters are saved in `.\models\teacher_net`.

**Second**, we should produce the soft targets of training set by running `python infer.py --phase teacher --temp 4.0`.  We can set different `temp` hyperparameters to generate different soft targets, since different temperature generated soft targets have different impacts on training small network. Since we need keep the same order of training data and soft targets, we construct the new dataset for student network training in `./data/mnist_soft_{temp}.npz`.
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

**Third**, we should train the student network with soft targets by running `python train.py --phase student --dropout 0.1 --stu_hsize 30 --temp 4.0 --use_soft True` . The temperature need to be the same as which used in teacher network. The softmax with temperature function is defined as follow:
```python
def softmax_with_temperature(logits, temp=1.0):
    logits_with_temp = logits/temp
    _softmax = fluid.layers.softmax(logits_with_temp)
    return _softmax
```
We define the hard loss and soft loss as:
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

The structure of student network is also a two layer fully connected network with fewer units each layer.
```python
def mlp_student(img, drop_prob, h_size):
    h1 = fluid.layers.fc(input=img, size=h_size, act='relu')
    drop1 = fluid.layers.dropout(h1, dropout_prob=drop_prob)
    h2 = fluid.layers.fc(input=drop1, size=h_size, act='relu')
    drop2 = fluid.layers.dropout(h2, dropout_prob=drop_prob)
    logits = fluid.layers.fc(input=drop2, size=10, act=None)
    return logits
```

**Four**, to verifythe effectiveness of using soft targets, we should also compare with results of training student without soft targets. We run `python train.py --phase student --stu_hsize 30 --drop_prob 0.1 ` and `python infer.py  --phase student --stu_hsize 30`  to get the testing performance without soft targets. The testing performance with soft targets is by running `python infer.py --phase student --stu_hsize 30 --use_soft True`.


## Results
We train the teacher network using SGD-Momentum optimizer with 0.001 learning rate for 200 epochs. The dropout ratio is 0.4.  The teacher network is the large network with 1200 units per fully connected layer. Finally, we get the testing accuracy 99.33%.
In order to show the effectiveness of knowledge distillation approach, we use a 30 units two layer fully connected network as the small network. The baseline student net is trained without soft targets by using SGD optimizer with 0.001 learning rate for 200 epochs. Finally, we get the testing accuracy 94.38%.
When we train the small network with soft targets of temperature 4.0, we get the testing accuracy 97.01%. It shows obvious improvement by using soft targets.
| Methods      |  Test Accuracy |  
| :-------- | --------:|
| TeacherNet    |  99.33% |
| StudentNet 30units without soft targets | 94.38%  |  
| StudentNet 30units with 4.0temp soft targets | 97.01%  |

The testing accuracy during training procedure are visualized here. We can see that using soft targets speed up the convergence.
![convergence](https://github.com/likesiwell/models/blob/distill-branch/distill_knowledge/images/plots.png)



## Reference
[^mnist]: [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)

[^distill]: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
