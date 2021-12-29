import paddle


def accuracy_paddle(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.equal(target)

        res = []
        for k in topk:
            correct_k = correct.astype(paddle.int32)[:k].flatten().sum(
                dtype='float32')
            res.append(correct_k / batch_size)
        return res
