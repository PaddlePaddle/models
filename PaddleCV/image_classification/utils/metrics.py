import paddle
import paddle.fluid as fluid


class Metrics(object):
    """A class used to generate metrics of specified model.

    Attributes:
    class_num:
    model_name:
    use_mixup:
    class_dim:
    use_label_smoothing:
    epsilon:
    model:
    image:
    label:
    is_train:
    
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

    def _clac_label_smoothing_loss(softmax_out, label, class_dim, epsilon):
        """Calculate label smoothing loss

            Returns:
            label smoothing loss
        
        """
        label_one_hot = fluid.layers.one_hot(input=label, depth=class_dim)
        smooth_label = fluid.layers.label_smoothing(
            label=label_one_hot, epsilon=epsilon, dtype="float32")
        loss = fluid.layers.cross_entropy(
            input=softmax_out, label=smooth_label, soft_label=True)

        return loss

    def out(self):
        """Default Metrics output, Include avg_cost, acc_top1 and acc_top5

            Returns:
            Metrics of default network
        
        """
        net_out = self.model.net(input=self.image, class_dim=self.class_dim)
        softmax_out = fluid.layers.softmax(net_out, use_cudnn=False)

        if self.is_train and self.use_label_smoothing:
            cost = _calc_label_smoothing_loss(softmax_out, self.label,
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
        
        Returns:
        Metrics of GoogLeNet
    """

    def __init__(self):
        super(GoogLeNet_Metrics, self).__init__()

    def out(self):
        """GoogLeNet Metrics output, Include avg_cost, acc_top1 and acc_top5


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

        return [avg_cost, avg_top1, avg_top5]


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
    """

    def __init__(self, data, model, args, is_train):
        super(Mixup_Metrics, self).__init__(data, model, args, is_train)
        #Instead of label, mixup use y_a y_b and lam to calculate metrics.
        self.y_a = data[1]
        self.y_b = data[2]
        self.lam = data[3]

    def out(self):
        """Metrics of Mixup processing network

        """

        net_out = self.model.net(input=self.image, class_dim=self.class_dim)
        softmax_out = fluid.layers.softmax(net_out, use_cudnn=False)

        if not self.use_label_smoothing:
            loss_a = fluid.layers.cross_entropy(
                input=softmax_out, label=self.y_a)
            loss_b = fluid.layers.cross_entropy(
                input=softmax_out, label=self.y_b)
        else:
            loss_a = Metrics._calc_label_smoothing(softmax_out, self.y_a,
                                                   self.class_dim, self.epsilon)
            loss_b = Metrics._calc_label_smoothing(softmax_out, self.y_b,
                                                   self.class_dim, self.epsilon)

        loss_a_mean = fluid.layers.mean(x=loss_a)
        loss_b_mean = fluid.layers.mean(x=loss_b)

        cost = self.lam * loss_a_mean + (1 - self.lam) * loss_b_mean
        avg_cost = fluid.layers.mean(x=cost)
        return [avg_cost]


def create_metrics(data, model, args, is_train):
    if args.model == "GoogLeNet":
        metrics = GoogLeNet_Metrics(data, model, args, is_train)
    else:
        if args.use_mixup and is_train:
            metrics = Mixup_Metrics(data, model, args, is_train)
        else:
            metrics = Metrics(data, model, args, is_train)
    return metrics
