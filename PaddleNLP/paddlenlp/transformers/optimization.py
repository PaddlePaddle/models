import paddle

__all__ = ['get_linear_schedule_with_warmup']


def get_linear_schedule_with_warmup(learning_rate,
                                    num_training_steps,
                                    num_warmup_steps,
                                    last_epoch=-1,
                                    verbose=False):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given initial learning rate, after this warmup period learning
    rate would be decreased linearly from the initial learning rate to 0.

    Args:
        learning_rate (float): The initial learning rate. It is a python float
            number.
        num_training_steps (int): The number of training steps.
        num_warmup_steps (int): The number of steps for warmup.
        last_epoch (int, optional): The index of last epoch. It can be set for
            resuming training. If None, it means initial learning rate. 
            Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for each
            update. Default: False.

    Returns:
        ``paddle.optimizer.lr.LambdaDecay`` instance to schedule learning rate
            with warmup linearly.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import get_linear_schedule_with_warmup
            lr, warmup_steps, max_steps = 0.1, 100, 1000
            lr_scheduler = get_linear_schedule_with_warmup(lr, 
                                                            warmup_steps, 
                                                            max_steps)

    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0,
                   float(num_training_steps - current_step) /
                   float(max(1, num_training_steps - num_warmup_steps)))

    return paddle.optimizer.lr.LambdaDecay(learning_rate, lr_lambda, last_epoch,
                                           verbose)
