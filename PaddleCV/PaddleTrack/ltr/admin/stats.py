

class StatValue:
    def __init__(self):
        self.clear()

    def reset(self):
        self.val = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val):
        self.val = val
        self.history.append(self.val)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.clear()
        self.has_new_data = False

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def new_epoch(self):
        if self.count > 0:
            self.history.append(self.avg)
            self.reset()
            self.has_new_data = True
        else:
            self.has_new_data = False


def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    single_input = not isinstance(topk, (tuple, list))
    if single_input:
        topk = (topk,)

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)[0]
        res.append(correct_k * 100.0 / batch_size)

    if single_input:
        return res[0]

    return res
