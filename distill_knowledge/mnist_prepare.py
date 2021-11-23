import paddle
import numpy as np
import os
reader = paddle.dataset.mnist.train()
img_list = []
label_list = []
for e in reader():
    img_list.append(e[0])
    label_list.append(e[1])

train_x = np.vstack(img_list)
train_y = np.vstack(label_list)
print(train_x.shape, train_y.shape)
print(np.min(train_x), np.max(train_x))

reader = paddle.dataset.mnist.test()
img_list = []
label_list = []
for e in reader():
    img_list.append(e[0])
    label_list.append(e[1])

test_x = np.vstack(img_list)
test_y = np.vstack(label_list)
print(test_x.shape, test_y.shape)

if not os.path.exits('./data/'):
    os.makedirs('./data/')
if not os.path.exits('./models/'):
    os.makedirs('./models')

np.savez(
    './data/mnist.npz',
    train_x=train_x,
    train_y=train_y,
    test_x=test_x,
    test_y=test_y)
