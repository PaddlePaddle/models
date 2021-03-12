from paddle.optimizer.lr import LRScheduler


class StepDecay(LRScheduler):
    """
    Set lr schedule in trainning process.
    """
    def __init__(self,
                learning_rate,
                step_size,
                warmup=0,
                gamma=0.1,
                last_epoch=-1,
                verbose=False):
        if not isinstance(step_size, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s." %
                type(step_size))
        if gamma >= 1.0:
            raise ValueError('gamma should be < 1.0.')

        self.warmup = warmup
        self.step_size = step_size
        self.gamma = gamma
        super(StepDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            i = 1
        else:
            i = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma**i)
