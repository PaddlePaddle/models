import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..', '..'))

from ltr.models.backbone.resnet_dilated import resnet50
from ltr.models.backbone.alexnet import AlexNet
from ltr.models.siam.head import DepthwiseRPN, MaskCorr, Refine
from ltr.models.siam.neck import AdjustAllLayer


class Siamnet(dygraph.layers.Layer):
    def __init__(self,
                 feature_extractor,
                 rpn_head,
                 neck=None,
                 mask_head=None,
                 refine_head=None,
                 scale_loss=None):

        super(Siamnet, self).__init__()

        self.feature_extractor = feature_extractor
        self.rpn_head = rpn_head
        self.neck = neck
        self.mask_head = mask_head
        self.refine_head = refine_head
        self.scale_loss = scale_loss

    def forward(self, template, search):
        # get feature
        if len(template.shape) == 5:
            template = fluid.layers.reshape(template, [-1, *list(template.shape)[-3:]])
            search = fluid.layers.reshape(search, [-1, *list(search.shape)[-3:]])
        
        zf = self.feature_extractor(template)
        xf = self.feature_extractor(search)
        if not self.mask_head is None:
            zf = zf[-1]
            xf_refine = xf[:-1]
            xf = xf[-1]
        if isinstance(zf, list):
            zf = zf[-1]
        if isinstance(xf, list):
            xf = xf[-1]
        if not self.neck is None:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        if not self.mask_head is None:
            if not self.refine_head is None:
                _, mask_corr_feature = self.mask_head(zf, xf)
                mask = self.refine_head(xf_refine, mask_corr_feature)
            else:
                mask, mask_corr_feature = self.mask_head(zf, xf)
            return {'cls': cls,
                    'loc': loc,
                    'mask': mask}
        else:
            return {'cls': cls,
                    'loc': loc}

    def extract_backbone_features(self, im):
        return self.feature_extractor(im)

    def template(self, template):
        zf = self.feature_extractor(template)
        if not self.mask_head is None:
            zf = zf[-1]
        if isinstance(zf, list):
            zf = zf[-1]
        if not self.neck is None:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, search):
        xf = self.feature_extractor(search)
        if not self.mask_head is None:
            self.xf = xf[:-1]
            xf = xf[-1]
        if isinstance(xf, list):
            xf = xf[-1]
        if not self.neck is None:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)

        if not self.mask_head is None:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
            return {'cls': cls,
                    'loc': loc,
                    'mask': mask}
        else:
            return {'cls': cls,
                    'loc': loc}

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos, test=True)


def SiamRPN_AlexNet(backbone_pretrained=True,
                    backbone_is_test=True,
                    is_test=False,
                    scale_loss=None):
    backbone = AlexNet(
        'AlexNet',
        is_test=backbone_is_test,
        output_layers=['conv5'],
        pretrained=backbone_pretrained)

    rpn_head = DepthwiseRPN(anchor_num=5, in_channels=256, out_channels=256, is_test=is_test)

    model = Siamnet(
        feature_extractor=backbone,
        rpn_head=rpn_head,
        scale_loss=scale_loss)
    return model


def SiamRPN_ResNet50(backbone_pretrained=True,
                     backbone_is_test=True,
                     is_test=False,
                     scale_loss=None):
    backbone = resnet50(
        'ResNet50',
        pretrained=backbone_pretrained,
        output_layers=[2],
        is_test=backbone_is_test)

    neck = AdjustAllLayer(in_channels=[1024], out_channels=[256], is_test=is_test)
    
    rpn_head = DepthwiseRPN(anchor_num=5, in_channels=256, out_channels=256, is_test=is_test)

    model = Siamnet(
        feature_extractor=backbone,
        neck=neck,
        rpn_head=rpn_head,
        scale_loss=scale_loss)
    return model


def SiamMask_ResNet50_base(backbone_pretrained=True,
                           backbone_is_test=True,
                           is_test=False,
                           scale_loss=None):
    backbone = resnet50(
        'ResNet50',
        pretrained=backbone_pretrained,
        output_layers=[0,1,2],
        is_test=backbone_is_test)
    
    neck = AdjustAllLayer(in_channels=[1024], out_channels=[256], is_test=is_test)
    
    rpn_head = DepthwiseRPN(anchor_num=5, in_channels=256, out_channels=256, is_test=is_test)

    mask_head = MaskCorr(in_channels=256, hidden=256, out_channels=3969, is_test=is_test)

    model = Siamnet(
        feature_extractor=backbone,
        neck=neck,
        rpn_head=rpn_head,
        mask_head=mask_head,
        scale_loss=scale_loss)
    return model


def SiamMask_ResNet50_sharp(backbone_pretrained=False,
                            backbone_is_test=True,
                            is_test=False,
                            scale_loss=None):
    backbone = resnet50(
        'ResNet50',
        pretrained=backbone_pretrained,
        output_layers=[0,1,2],
        is_test=backbone_is_test)
    
    neck = AdjustAllLayer(in_channels=[1024], out_channels=[256], is_test=True)
    
    rpn_head = DepthwiseRPN(anchor_num=5, in_channels=256, out_channels=256, is_test=True)

    mask_head = MaskCorr(in_channels=256, hidden=256, out_channels=3969, is_test=is_test)

    refine_head = Refine()

    model = Siamnet(
        feature_extractor=backbone,
        neck=neck,
        rpn_head=rpn_head,
        mask_head=mask_head,
        refine_head=refine_head,
        scale_loss=scale_loss)
    return model


if __name__ == '__main__':
    import numpy as np

    search = np.random.uniform(-1, 1, [1, 3, 255, 255]).astype(np.float32)
    template = np.random.uniform(-1, 1, [1, 3, 127, 127]).astype(np.float32)
    with fluid.dygraph.guard():
        search = fluid.dygraph.to_variable(search)
        template = fluid.dygraph.to_variable(template)

        model = SiamMask(False)

        res = model(template, search)
        params = model.state_dict()
        for v in params:
            print(v)
