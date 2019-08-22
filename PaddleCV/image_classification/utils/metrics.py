#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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


class Metrics(object):
    """A class used to generate metrics of specified model.

    Attributes:

        model_name: the name of model
        use_mixup: whether to use mixup
        class_dim: the number of class
        use_label_smoothing: whether to use label_smoothing
        epsilon: the label_smoothing epsilon
        model: class of the model architecture
        image: iamge data
        label: label data
        is_train: mode
    
    """

    def __init__(self, data, model, args, is_train):

        self.model_name = args.model
        self.use_mixup = args.use_mixup
        self.class_dim = args.class_dim
        self.use_label_smoothing = args.use_label_smoothing
        self.epsilon = args.label_smoothing_epsilon
        self.model = model
        self.image = data[0]
        self.label = data[1]
        self.is_train = is_train

    def _calc_label_smoothing_loss(self, softmax_out, label, class_dim,
                                   epsilon):
        """Calculate label smoothing loss

        Returns:
            label smoothing loss
        
        """
        label_one_hot = fluid.layers.one_hot(input=label, depth=class_dim)
        smooth_label = fluid.layers.label_smooth(
            label=label_one_hot, epsilon=epsilon, dtype="float32")
        loss = fluid.layers.cross_entropy(
            input=softmax_out, label=smooth_label, soft_label=True)

        return loss

    def out(self):
        """Default Metrics output, Include avg_cost, acc_top1 and acc_top5

        Returns:
            Metrics of default net
        
        """
        net_out = self.model.net(input=self.image, class_dim=self.class_dim)
        softmax_out = fluid.layers.softmax(net_out, use_cudnn=False)

        if self.is_train and self.use_label_smoothing:
            cost = self._calc_label_smoothing_loss(softmax_out, self.label,
                                                   self.class_dim, self.epsilon)

        else:
            cost = fluid.layers.cross_entropy(
                input=softmax_out, label=self.label)

        avg_cost = fluid.layers.mean(cost)
        acc_top1 = fluid.layers.accuracy(
            input=softmax_out, label=self.label, k=1)
        acc_top5 = fluid.layers.accuracy(
            input=softmax_out, label=self.label, k=5)

        return [avg_cost, acc_top1, acc_top5]


class GoogLeNet_Metrics(Metrics):
    """A subclass inherited from Metrics
    """

    def __init__(self, data, model, args, is_train):
        super(GoogLeNet_Metrics, self).__init__(data, model, args, is_train)

    def out(self):
        """GoogLeNet Metrics output, include avg_cost, acc_top1 and acc_top5
        
        Returns:
            Metrics of GoogLeNet

        """
        out0, out1, out2 = self.model.net(input=self.image,
                                          class_dim=self.class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=self.label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=self.label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=self.label)

        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=self.label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=self.label, k=5)

        return [avg_cost, acc_top1, acc_top5]


#TODO: (2019/08/08) Distill is temporary disabled now.
"""
class Distill_Metrics(Metrics):
    def __init__(self):
        super(Distill_Metrics,self).__init__()
    def out(self):
        out1, out2 = self.model.net(input=self.image, class_dim=self.class_dim)
        softmax_out1, softmax_out = fluid.layers.softmax(out1), fluid.layers.softmax(out2)
        smooth_out1 = fluid.layers.label_smooth(label=softmax_out1, epsilon=0.0, dtype="float32")
        cost = fluid.layers.cross_entropy(input=softmax_out, label=smooth_out1, soft_label=True)
        avg_cost = fluid.layers.mean(cost)

        return avg_cost
"""


class Mixup_Metrics(Metrics):
    """A subclass inherited from Metrics

        Note: Mixup preprocessing only apply on the training process.
        
        Attributes:
            y_a: label
            y_b: label
            lam: lamda
    """

    def __init__(self, data, model, args, is_train):
        super(Mixup_Metrics, self).__init__(data, model, args, is_train)
        self.y_a = data[1]
        self.y_b = data[2]
        self.lam = data[3]

    def out(self):
        """Metrics of Mixup processing netw ork, include avg_cost
        """

        net_out = self.model.net(input=self.image, class_dim=self.class_dim)
        softmax_out = fluid.layers.softmax(net_out, use_cudnn=False)
        if not self.use_label_smoothing:
            loss_a = fluid.layers.cross_entropy(
                input=softmax_out, label=self.y_a)
            loss_b = fluid.layers.cross_entropy(
                input=softmax_out, label=self.y_b)
        else:
            loss_a = Metrics._calc_label_smoothing_loss(
                self, softmax_out, self.y_a, self.class_dim, self.epsilon)
            loss_b = Metrics._calc_label_smoothing_loss(
                self, softmax_out, self.y_b, self.class_dim, self.epsilon)

        loss_a_mean = fluid.layers.mean(x=loss_a)
        loss_b_mean = fluid.layers.mean(x=loss_b)
        cost = self.lam * loss_a_mean + (1 - self.lam) * loss_b_mean
        avg_cost = fluid.layers.mean(x=cost)
        return [avg_cost]


def create_metrics(data, model, args, is_train):
    """Create metrics, include GoogLeNet(train, test); mixup(train); default(train, test) metrics
    """
    if args.model == "GoogLeNet":
        metrics = GoogLeNet_Metrics(data, model, args, is_train)
    else:
        if args.use_mixup and is_train:
            metrics = Mixup_Metrics(data, model, args, is_train)
        else:
            metrics = Metrics(data, model, args, is_train)
    return metrics
