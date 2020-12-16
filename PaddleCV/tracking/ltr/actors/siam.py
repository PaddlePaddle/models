from . import BaseActor
import paddle.fluid as fluid
import numpy as np

class SiamActor(BaseActor):
    """ Actor for training the SiamRPN/SiamMask"""

    def __call__(self, data):
        # Run network to obtain predictiion
        pred = self.net(data['train_images'], data['test_images'])

        # Compute loss
        label_cls = fluid.layers.cast(x=data['label_cls'], dtype=np.int64)
        cls_loss = self.objective['cls'](pred['cls'], label_cls)
        loc_loss = self.objective['loc'](pred['loc'], data['label_loc'], data['label_loc_weight'])
        
        loss = {}
        loss['cls'] = cls_loss
        loss['loc'] = loc_loss
        
        # Return training stats
        stats = {}
        stats['Loss/cls'] = cls_loss.numpy()
        stats['Loss/loc'] = loc_loss.numpy()

        # Compute mask loss if necessary
        if 'mask' in pred:
            mask_loss, iou_m, iou_5, iou_7 = self.objective['mask'](
                pred['mask'],
                data['label_mask'],
                data['label_mask_weight'])
            loss['mask'] = mask_loss
            
            stats['Loss/mask'] = mask_loss.numpy()
            stats['Accuracy/mask_iou_mean'] = iou_m.numpy()
            stats['Accuracy/mask_at_5'] = iou_5.numpy()
            stats['Accuracy/mask_at_7'] = iou_7.numpy()

        # Use scale loss if exists
        scale_loss = getattr(self.net, "scale_loss", None)
        if callable(scale_loss):
            total_loss = scale_loss(loss)
        else:
            total_loss = 0
            for k, v in loss.items():
                total_loss += v
        
        stats['Loss/total'] = total_loss.numpy()

        return total_loss, stats
