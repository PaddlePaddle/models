import paddle
import paddle.nn.functional as F
import numpy as np

class MixUpLoss(paddle.nn.Layer):
     
    def __init__(self,criterion):
        super(MixUpLoss, self).__init__()
        self.criterion = criterion

    def forward(self,pred,mixup_target):
        assert type(mixup_target) in [tuple,list] and len(mixup_target)==3,'mixup data should be tuple consists of (ya,yb,lamda)'
        ya,yb,lamda = mixup_target
        return lamda * self.criterion(pred, ya) \
                + (1 - lamda) * self.criterion(pred, yb)
    
    def extra_repr(self):
        return 'MixUpLoss with {}'.format(self.criterion)




def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = paddle.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) *  paddle.index_select(x,index)#xb# x[index, :]
    y_a, y_b = y, paddle.index_select(y,index)#paddle.concat([y[int(i):int(i+1)] for i in index])# y[index]
    mixed_target = (y_a, y_b, lam)
    return mixed_x, mixed_target
