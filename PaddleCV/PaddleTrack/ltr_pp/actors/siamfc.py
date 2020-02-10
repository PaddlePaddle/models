import numpy as np
import paddle.fluid as fluid

from . import BaseActor


class SiamFCActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""

    def __init__(self, net, objective, batch_size, shape, radius, stride):
        super().__init__(net, objective)
        self.label_mask, self.label_weights = self._creat_gt_mask(batch_size, shape, radius, stride)

    def _creat_gt_mask(self, batch_size, shape, radius, stride):
        h, w = shape
        y = np.arange(h, dtype=np.float32) - (h - 1) / 2.
        x = np.arange(w, dtype=np.float32) - (w - 1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x ** 2 + y ** 2)
        mask = np.zeros((h, w))
        mask[dist <= radius / stride] = 1
        mask = mask[np.newaxis, :, :]
        weights = np.ones_like(mask)
        weights[mask == 1] = 0.5 / np.sum(mask == 1)
        weights[mask == 0] = 0.5 / np.sum(mask == 0)
        mask = np.repeat(mask, batch_size, axis=0)[:, np.newaxis, :, :]
        weights = np.repeat(weights, batch_size, axis=0)[:, np.newaxis, :, :]
        weights = fluid.dygraph.to_variable(weights.astype(np.float32))
        mask = fluid.dygraph.to_variable(mask.astype(np.float32))
        return mask, weights

    def __call__(self, data):
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        target_estimations = self.net(data['train_images'], data['test_images'])

        # weighted loss
        loss_mat = fluid.layers.sigmoid_cross_entropy_with_logits(target_estimations,
                                                                  self.label_mask,
                                                                  normalize=False)
        loss = fluid.layers.elementwise_mul(loss_mat, self.label_weights)
        loss = fluid.layers.reduce_sum(loss) / loss.shape[0]

        # Return training stats
        stats = {'Loss/total': loss.numpy(),
                 'Loss/center': loss.numpy()}

        return loss, stats
