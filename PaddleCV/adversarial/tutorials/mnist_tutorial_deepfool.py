#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DeepFool tutorial on mnist using advbox tool.
Deepfool is a simple and accurate adversarial attack method.
It supports both targeted attack and non-targeted attack.
"""
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import paddle.fluid as fluid
import paddle

from advbox.adversary import Adversary
from advbox.attacks.deepfool import DeepFoolAttack
from advbox.models.paddle import PaddleModel
from tutorials.mnist_model import mnist_cnn_model


def main():
    """
    Advbox demo which demonstrate how to use advbox.
    """
    TOTAL_NUM = 500
    IMG_NAME = 'img'
    LABEL_NAME = 'label'

    img = fluid.layers.data(name=IMG_NAME, shape=[1, 28, 28], dtype='float32')
    # gradient should flow
    img.stop_gradient = False
    label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')
    logits = mnist_cnn_model(img)
    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # use CPU
    place = fluid.CPUPlace()
    # use GPU
    # place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    BATCH_SIZE = 1
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.test(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    fluid.io.load_params(
        exe, "./mnist/", main_program=fluid.default_main_program())

    # advbox demo
    m = PaddleModel(
        fluid.default_main_program(),
        IMG_NAME,
        LABEL_NAME,
        logits.name,
        avg_cost.name, (-1, 1),
        channel_axis=1)
    attack = DeepFoolAttack(m)
    attack_config = {"iterations": 100, "overshoot": 9}

    # use train data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    for data in train_reader():
        total_count += 1
        adversary = Adversary(data[0][0], data[0][1])

        # DeepFool non-targeted attack
        adversary = attack(adversary, **attack_config)

        # DeepFool targeted attack
        # tlabel = 0
        # adversary.set_target(is_targeted_attack=True, target_label=tlabel)
        # adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (data[0][1], adversary.adversarial_label, total_count))
            # plt.imshow(adversary.target, cmap='Greys_r')
            # plt.show()
            # np.save('adv_img', adversary.target)
        else:
            print('attack failed, original_label=%d, count=%d' %
                  (data[0][1], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TRAIN_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break

    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    for data in test_reader():
        total_count += 1
        adversary = Adversary(data[0][0], data[0][1])

        # DeepFool non-targeted attack
        adversary = attack(adversary, **attack_config)

        # DeepFool targeted attack
        # tlabel = 0
        # adversary.set_target(is_targeted_attack=True, target_label=tlabel)
        # adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (data[0][1], adversary.adversarial_label, total_count))
            # plt.imshow(adversary.target, cmap='Greys_r')
            # plt.show()
            # np.save('adv_img', adversary.target)
        else:
            print('attack failed, original_label=%d, count=%d' %
                  (data[0][1], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break
    print("deelfool attack done")


if __name__ == '__main__':
    main()
