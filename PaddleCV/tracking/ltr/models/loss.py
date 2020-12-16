import paddle.fluid as fluid
import numpy as np


def get_cls_loss(pred, label, select):
    if select.shape[0] == 0:
        return fluid.layers.reduce_sum(pred) * 0
    pred = fluid.layers.gather(pred, select)
    label = fluid.layers.gather(label, select)
    label = fluid.layers.reshape(label, [-1, 1])
    loss =  fluid.layers.softmax_with_cross_entropy(
        logits = pred, 
        label = label)
    return fluid.layers.mean(loss)


def select_softmax_with_cross_entropy_loss(pred, label):
    b, c, h, w = pred.shape
    pred = fluid.layers.reshape(pred, [b, 2, -1, h, w])
    pred = fluid.layers.transpose(pred, [0, 2, 3, 4, 1])
    pred = fluid.layers.reshape(pred, [-1, 2])
    label = fluid.layers.reshape(label, [-1])
    pos = fluid.layers.where(label == 1)
    neg = fluid.layers.where(label == 0)
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, c, h, w = pred_loc.shape
    pred_loc = fluid.layers.reshape(pred_loc, [b, 4, -1, h, w])
    loss = fluid.layers.abs(pred_loc - label_loc)
    loss = fluid.layers.reduce_sum(loss, dim=1)
    loss = loss * loss_weight
    return fluid.layers.reduce_sum(loss) / b


def soft_margin_loss(pred, label):
    #loss = fluid.layers.elementwise_mul(pred, label)
    loss = fluid.layers.exp(-1 * pred * label)
    loss = fluid.layers.log(1 + loss)
    return fluid.layers.reduce_mean(loss)


def iou_measure(pred, label):
    pred = fluid.layers.cast(pred >= 0, 'float32')
    pred = fluid.layers.cast(pred == 1, 'float32')
    label = fluid.layers.cast(label == 1, 'float32')
    mask_sum = pred + label
    intxn = fluid.layers.reduce_sum(
        fluid.layers.cast(mask_sum == 2, 'float32'), dim=1)
    union = fluid.layers.reduce_sum(
        fluid.layers.cast(mask_sum > 0, 'float32'), dim=1)
    iou = intxn / union
    iou_m = fluid.layers.reduce_mean(iou)
    iou_5 = fluid.layers.cast(iou > 0.5, 'float32')
    iou_5 = fluid.layers.reduce_sum(iou_5) / iou.shape[0]
    iou_7 = fluid.layers.cast(iou > 0.7, 'float32')
    iou_7 = fluid.layers.reduce_sum(iou_7) / iou.shape[0]
    return iou_m, iou_5, iou_7
    

def select_mask_logistic_loss(pred_mask, label_mask, loss_weight, out_size=63, gt_size=127):
    loss_weight = fluid.layers.reshape(loss_weight, [-1])
    pos = loss_weight == 1
    if np.sum(pos.numpy()) == 0:
        return fluid.layers.reduce_sum(pred_mask) * 0, fluid.layers.reduce_sum(pred_mask) * 0, fluid.layers.reduce_sum(pred_mask) * 0, fluid.layers.reduce_sum(pred_mask) * 0
    pos = fluid.layers.where(pos)
    if len(pred_mask.shape) == 4:
        pred_mask = fluid.layers.transpose(pred_mask, [0, 2, 3, 1])
        pred_mask = fluid.layers.reshape(pred_mask, [-1, 1, out_size, out_size])
        pred_mask = fluid.layers.gather(pred_mask, pos)
        pred_mask = fluid.layers.resize_bilinear(pred_mask, out_shape=[gt_size, gt_size]);
        pred_mask = fluid.layers.reshape(pred_mask, [-1, gt_size * gt_size])
        label_mask_uf = fluid.layers.unfold(label_mask, [gt_size, gt_size], 8, 32)
    else:
        pred_mask = fluid.layers.gather(pred_mask, pos)
        label_mask_uf = fluid.layers.unfold(label_mask, [gt_size, gt_size], 8, 0)

    label_mask_uf = fluid.layers.transpose(label_mask_uf, [0, 2, 1])
    label_mask_uf = fluid.layers.reshape(label_mask_uf, [-1, gt_size * gt_size])

    label_mask_uf = fluid.layers.gather(label_mask_uf, pos)
    loss = soft_margin_loss(pred_mask, label_mask_uf)
    if np.isnan(loss.numpy()):
        return fluid.layers.reduce_sum(pred_mask) * 0, fluid.layers.reduce_sum(pred_mask) * 0, fluid.layers.reduce_sum(pred_mask) * 0, fluid.layers.reduce_sum(pred_mask) * 0
    iou_m, iou_5, iou_7 = iou_measure(pred_mask, label_mask_uf)
    return loss, iou_m, iou_5, iou_7


if __name__ == "__main__":
    import numpy as np
    pred_mask = np.random.randn(4, 63*63, 25, 25)
    weight_mask = np.random.randn(4, 1, 25, 25) > 0.9
    label_mask = np.random.randint(-1, 1, (4, 1, 255, 255))

    pred_loc = np.random.randn(3, 32, 17, 17)
    weight_loc = np.random.randn(3, 8, 17, 17)
    label_loc = np.random.randn(3, 4, 8, 17, 17)

    pred_cls = np.random.randn(3, 16, 17, 17)
    label_cls = np.random.randint(0, 2, (3, 8, 17, 17))

    with fluid.dygraph.guard():
        pred_mask = fluid.dygraph.to_variable(pred_mask)
        weight_mask = fluid.dygraph.to_variable(weight_mask.astype('float32'))
        label_mask = fluid.dygraph.to_variable(label_mask.astype('float32'))
        loss = select_mask_logistic_loss(pred_mask, label_mask, weight_mask)
        print("loss_mask = ", loss)

        pred_loc = fluid.dygraph.to_variable(pred_loc)
        weight_loc = fluid.dygraph.to_variable(weight_loc)
        label_loc = fluid.dygraph.to_variable(label_loc)
        loss = weight_l1_loss(pred_loc, label_loc, weight_loc)
        print("loss_loc = ", loss)

        pred_cls = fluid.dygraph.to_variable(pred_cls)
        label_cls = fluid.dygraph.to_variable(label_cls)
        loss = select_softmax_with_cross_entropy_loss(pred_cls, label_cls)
        print("loss_cls = ", loss)
