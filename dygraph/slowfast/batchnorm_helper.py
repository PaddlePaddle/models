from functools import partial
import paddle


def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Layer: the normalization layer.
    """
    if cfg.TRAIN.bn_norm_type == "batchnorm":
        return paddle.nn.BatchNorm3D
    elif cfg.TRAIN.bn_norm_type == "sub_batchnorm":
        return partial(SubBatchNorm3D, num_splits=cfg.TRAIN.bn_num_splits)
    else:
        raise NotImplementedError("Norm type {} is not supported".format(
            cfg.TRAIN.bn_norm_type))


def aggregate_sub_bn_stats(model):
    """
    Recursively find all SubBN modules and aggregate sub-BN stats.
    Args:
        model (nn.Layer): model to be aggregate sub-BN stats
    Returns:
        count (int): number of SubBN module found.
    """
    count = 0
    for child in model.children():
        if isinstance(child, SubBatchNorm3D):
            child.aggregate_stats()
            count += 1
        else:
            count += aggregate_sub_bn_stats(child)
    return count


class SubBatchNorm3D(paddle.nn.Layer):
    """
    Implement based on paddle2.0.
    The standard BN layer computes stats across all examples in a GPU. In some
    cases it is desirable to compute stats across only a subset of examples
    SubBatchNorm3D splits the batch dimension into N splits, and run BN on
    each of them separately (so that the stats are computed on each subset of
    examples (1/N of batch) independently. During evaluation, it aggregates
    the stats from all splits into one BN.
    """

    def __init__(self, num_splits, **args):
        """
        Args:
            num_splits (int): number of splits.
            args (list): list of args
        """
        super(SubBatchNorm3D, self).__init__()
        self.num_splits = num_splits
        self.num_features = args["num_features"]
        self.weight_attr = args["weight_attr"]
        self.bias_attr = args["bias_attr"]

        # Keep only one set of weight and bias (outside).
        if self.weight_attr == False:
            self.weight = self.create_parameter(
                attr=None,
                shape=[self.num_features],
                default_initializer=paddle.nn.initializer.Constant(1.0))
            self.weight.stop_gradient = True
        else:
            self.weight = self.create_parameter(
                attr=self.weight_attr,
                shape=[self.num_features],
                default_initializer=paddle.nn.initializer.Constant(1.0))
            self.weight.stop_gradient = self.weight_attr != None \
                                        and self.weight_attr.learning_rate == 0.

        if self.bias_attr == False:
            self.bias = self.create_parameter(
                attr=None, shape=[self.num_features], is_bias=True)
            self.bias.stop_gradient = True
        else:
            self.bias = self.create_parameter(
                attr=self.bias_attr, shape=[self.num_features], is_bias=True)
            self.bias.stop_gradient = self.bias_attr != None \
                                      and self.bias_attr.learning_rate == 0.

        # set weights and bias fixed (inner).
        args["weight_attr"] = False
        args["bias_attr"] = False
        self.bn = paddle.nn.BatchNorm3D(**args)
        # update number of features used in split_bn
        args["num_features"] = self.num_features * self.num_splits
        self.split_bn = paddle.nn.BatchNorm3D(**args)

    def _get_aggregated_mean_std(self, means, stds, n):
        """
        Calculate the aggregated mean and stds.
        Use the method of update mean and std when merge multi-part data.
        Args:
            means (tensor): mean values.
            stds (tensor): standard deviations.
            n (int): number of sets of means and stds.
        """
        mean = paddle.sum(paddle.reshape(means, (n, -1)), axis=0) / n
        std = (paddle.sum(paddle.reshape(stds, (n, -1)), axis=0) / n +
               paddle.sum(paddle.reshape(
                   paddle.pow((paddle.reshape(means, (n, -1)) - mean), 2),
                   (n, -1)),
                          axis=0) / n)
        return mean, std

    def aggregate_stats(self):
        """
        Synchronize running_mean, and running_var to self.bn.
        Call this before eval, then call model.eval();
        When eval, forward function will call self.bn instead of self.split_bn,
        During this time the running_mean, and running_var of self.bn has been obtained from
        self.split_bn.
        """
        if self.split_bn.training:
            bn_mean_tensor, bn_variance_tensor = self._get_aggregated_mean_std(
                self.split_bn._mean,
                self.split_bn._variance,
                self.num_splits, )
            self.bn._mean.set_value(bn_mean_tensor)
            self.bn._variance.set_value(bn_variance_tensor)

    def forward(self, x):
        if self.training:
            n, c, t, h, w = x.shape
            x = paddle.reshape(x, (n // self.num_splits, c * self.num_splits, t,
                                   h, w))
            x = self.split_bn(x)
            x = paddle.reshape(x, (n, c, t, h, w))
        else:
            x = self.bn(x)
        x = paddle.multiply(x, paddle.reshape(self.weight, (-1, 1, 1, 1)))
        x = paddle.add(x, paddle.reshape(self.bias, (-1, 1, 1, 1)))
        return x
