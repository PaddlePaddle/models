from paddle import fluid
from paddle.fluid import dygraph
import ltr_pp.models.siamese.target_estimator_net as tgt_estimator


class SiamNet(dygraph.layers.Layer):
    def __init__(self, name, feature_extractor, target_estimator, target_estimator_layer, extractor_grad=True):
        """

        :param feature_extractor: backbone
        :param target_estimator: headers
        :param target_estimator_layer: list, which layer is used in header,
        :param extractor_grad: default is True
        """
        super(SiamNet, self).__init__(name)

        self.feature_extractor = feature_extractor
        self.target_estimator = target_estimator
        self.target_estimator_layer = target_estimator_layer

    def forward(self, train_imgs, test_imgs):
        # extract backbone features
        if len(train_imgs.shape) == 5:
            train_imgs = fluid.layers.reshape(train_imgs, [-1, *list(train_imgs.shape)[-3:]])
            test_imgs = fluid.layers.reshape(test_imgs, [-1, *list(test_imgs.shape)[-3:]])

        train_feat = self.extract_backbone_features(train_imgs)
        test_feat = self.extract_backbone_features(test_imgs)

        train_feat = [feat for feat in train_feat.values()]
        test_feat = [feat for feat in test_feat.values()]

        # Obtain target estimation
        targets = self.target_estimator(train_feat, test_feat)
        return targets

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.target_estimator_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)


def siamfc_alexnet(backbone_pretrained=False, backbone_is_test=False, estimator_is_test=False):
    from ltr_pp.models.backbone.sfc_alexnet import SFC_AlexNet
    backbone_net = SFC_AlexNet('AlexNet', is_test=backbone_is_test)
    target_estimator = tgt_estimator.SiamFCEstimator('CenterEstimator')
    model = SiamNet('SiamFC', backbone_net, target_estimator, ['conv5'], )
    return model
