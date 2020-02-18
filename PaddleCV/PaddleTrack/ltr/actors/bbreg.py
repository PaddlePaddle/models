from . import BaseActor
import paddle.fluid as fluid


class AtomActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        iou_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        iou_pred = fluid.layers.reshape(iou_pred, [-1, iou_pred.shape[2]])
        iou_gt = fluid.layers.reshape(data['proposal_iou'], [-1, data['proposal_iou'].shape[2]])

        # Compute loss
        loss = self.objective(iou_pred, iou_gt)
        loss = fluid.layers.mean(loss)

        # Use scale loss if exists
        scale_loss = getattr(self.net, "scale_loss", None)
        if callable(scale_loss):
            loss = scale_loss(loss)

        # Return training stats
        stats = {'Loss/total': loss.numpy(),
                 'Loss/iou': loss.numpy()}

        return loss, stats
