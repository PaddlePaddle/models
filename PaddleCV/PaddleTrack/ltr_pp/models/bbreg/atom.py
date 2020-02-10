import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..', '..'))

from ltr_pp.models.backbone.resnet import resnet50, resnet18
from ltr_pp.models.bbreg.atom_iou_net import AtomIouNet


class ATOMnet(dygraph.layers.Layer):
    def __init__(self, name, feature_extractor, bb_regressor,
                 bb_regressor_layer, extractor_grad=True):
        """

        :param feature_extractor: backbone
        :param bb_regressor: IOUnet
        :param bb_regressor_layer: list, which layer is used in IOUnet,
        :param extractor_grad: default is True
        """
        super(ATOMnet, self).__init__(name)

        self.feature_extractor = feature_extractor
        self.bb_regressor = bb_regressor
        self.bb_regressor_layer = bb_regressor_layer

        layers_gt = ['block0', 'block1', 'block2', 'block3', 'fc']
        ##  pytorch [layer1, layer2, layer3, layer4]
        if bb_regressor_layer is not None:
            for key in bb_regressor_layer:
                assert key in layers_gt
        else:
            raise ValueError("bb_regressor_layer can only be one of :", layers_gt)

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals):
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if len(train_imgs.shape) == 5 else 1
        num_test_images = test_imgs.shape[0] if len(test_imgs.shape) == 5 else 1

        if len(train_imgs.shape) == 5:
            train_imgs = fluid.layers.reshape(train_imgs, [-1, *list(train_imgs.shape)[-3:]])
            test_imgs = fluid.layers.reshape(test_imgs, [-1, *list(test_imgs.shape)[-3:]])

        train_feat = self.extract_backbone_features(train_imgs)
        test_feat = self.extract_backbone_features(test_imgs)

        # For clarity, send the features to bb_regressor in sequenceform, i.e. [sequence, batch, feature, row, col]
        train_feat_iou = [fluid.layers.reshape(feat, (num_train_images, num_sequences, *feat.shape[-3:])) for feat in
                          train_feat.values()]
        test_feat_iou = [fluid.layers.reshape(feat, (num_test_images, num_sequences, *feat.shape[-3:])) for feat in
                         test_feat.values()]

        # Obtain iou prediction
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)
        return iou_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)


def atom_resnet18(iou_input_dim=(256, 256), iou_inter_dim=(256, 256), backbone_pretrained=True,
                  backbone_is_test=False, iounet_is_test=False):
    backbone = resnet18('ResNet18', is_test=backbone_is_test, pretrained=backbone_pretrained)
    iou_predictor = AtomIouNet('IOUnet',
                               pred_input_dim=iou_input_dim,
                               pred_inter_dim=iou_inter_dim,
                               is_test=iounet_is_test)

    model = ATOMnet('ATOM', feature_extractor=backbone,
                    bb_regressor=iou_predictor,
                    bb_regressor_layer=['block1', 'block2'],
                    extractor_grad=False)
    return model


def atom_resnet50(iou_input_dim=(256, 256), iou_inter_dim=(256, 256), backbone_pretrained=True,
                  backbone_is_test=False, iounet_is_test=False):
    backbone = resnet50('ResNet50', is_test=backbone_is_test, pretrained=backbone_pretrained)
    iou_predictor = AtomIouNet('IOUnet',
                               input_dim=(512, 1024),
                               pred_input_dim=iou_input_dim,
                               pred_inter_dim=iou_inter_dim,
                               is_test=iounet_is_test)

    model = ATOMnet('ATOM', feature_extractor=backbone,
                    bb_regressor=iou_predictor,
                    bb_regressor_layer=['block1', 'block2'],
                    extractor_grad=False)
    return model


if __name__ == '__main__':
    import numpy as np

    a = np.random.uniform(-1, 1, [1, 3, 144, 144]).astype(np.float32)
    b = np.random.uniform(-1, 1, [1, 3, 144, 144]).astype(np.float32)
    bbox = [[3, 4, 10, 11]]
    proposal_bbox = [[4, 5, 11, 12] * 16]
    bbox = np.reshape(np.array(bbox), [1, 1, 4]).astype(np.float32)
    proposal_bbox = np.reshape(np.array(proposal_bbox), [1, 16, 4]).astype(np.float32)
    with fluid.dygraph.guard():
        a_pd = fluid.dygraph.to_variable(a)
        b_pd = fluid.dygraph.to_variable(b)
        bbox_pd = fluid.dygraph.to_variable(bbox)
        proposal_bbox_pd = fluid.dygraph.to_variable(proposal_bbox)

        model = atom_resnet50()

        res = model(a_pd, b_pd, bbox_pd, proposal_bbox_pd)
        params = model.state_dict()
        for v in params:
            print(v)
