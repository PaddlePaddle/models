import os

import numpy as np
from paddle import fluid

from ltr.models.bbreg.atom import atom_resnet50, atom_resnet18
from ltr.models.siamese.siam import siamfc_alexnet
from ltr.models.siam.siam import SiamRPN_AlexNet, SiamMask_ResNet50_sharp, SiamMask_ResNet50_base
from pytracking.admin.environment import env_settings
from pytracking.features.featurebase import MultiFeatureBase
from pytracking.libs import TensorList
from pytracking.libs.paddle_utils import n2p


class ResNet18(MultiFeatureBase):
    """ResNet18 feature.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """

    def __init__(self,
                 output_layers=('block2', ),
                 net_path='atom_iou',
                 use_gpu=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.output_layers = list(output_layers)
        self.use_gpu = use_gpu
        self.net_path = net_path

    def initialize(self):
        with fluid.dygraph.guard():
            if os.path.isabs(self.net_path):
                net_path_full = self.net_path
            else:
                net_path_full = os.path.join(env_settings().network_path,
                                             self.net_path)

            self.net = atom_resnet18(
                backbone_pretrained=False,
                backbone_is_test=True,
                iounet_is_test=True)

            state_dictsm, _ = fluid.load_dygraph(net_path_full)
            self.net.load_dict(state_dictsm)
            self.net.train()

            self.iou_predictor = self.net.bb_regressor

        self.layer_stride = {
            'conv0': 2,
            'conv1': 2,
            'block0': 4,
            'block1': 8,
            'block2': 16,
            'block3': 32,
            'classification': 16,
            'fc': None
        }
        self.layer_dim = {
            'conv0': 64,
            'conv1': 64,
            'block0': 64,
            'block1': 128,
            'block2': 256,
            'block3': 512,
            'classification': 256,
            'fc': None
        }

        self.iounet_feature_layers = self.net.bb_regressor_layer

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1] * len(self.output_layers)

        self.feature_layers = sorted(
            list(set(self.output_layers + self.iounet_feature_layers)))

        self.mean = np.reshape([0.485, 0.456, 0.406], [1, -1, 1, 1])
        self.std = np.reshape([0.229, 0.224, 0.225], [1, -1, 1, 1])

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net
        if hasattr(self, 'iou_predictor'):
            del self.iou_predictor
        if hasattr(self, 'iounet_backbone_features'):
            del self.iounet_backbone_features
        if hasattr(self, 'iounet_features'):
            del self.iounet_features

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([
            s * self.layer_stride[l]
            for l, s in zip(self.output_layers, self.pool_stride)
        ])

    def extract(self, im: np.ndarray, debug_save_name=None):
        with fluid.dygraph.guard():
            if debug_save_name is not None:
                np.savez(debug_save_name, im)

            im = im / 255.  # don't use im /= 255. since we don't want to alter the input
            im -= self.mean
            im /= self.std
            im = n2p(im)

            output_features = self.net.extract_features(im, self.feature_layers)

            # Store the raw resnet features which are input to iounet
            iounet_backbone_features = TensorList([
                output_features[layer] for layer in self.iounet_feature_layers
            ])
            self.iounet_backbone_features = iounet_backbone_features.numpy()

            # Store the processed features from iounet, just before pooling
            self.iounet_features = TensorList([
                f.numpy()
                for f in self.iou_predictor.get_iou_feat(
                    iounet_backbone_features)
            ])

            output = TensorList([
                output_features[layer].numpy() for layer in self.output_layers
            ])
            return output


class ResNet50(MultiFeatureBase):
    """ResNet50 feature.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """

    def __init__(self,
                 output_layers=('block2', ),
                 net_path='atom_iou',
                 use_gpu=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.output_layers = list(output_layers)
        self.use_gpu = use_gpu
        self.net_path = net_path

    def initialize(self):
        with fluid.dygraph.guard():
            if os.path.isabs(self.net_path):
                net_path_full = self.net_path
            else:
                net_path_full = os.path.join(env_settings().network_path,
                                             self.net_path)

            self.net = atom_resnet50(
                backbone_pretrained=False,
                backbone_is_test=True,
                iounet_is_test=True)

            state_dictsm, _ = fluid.load_dygraph(net_path_full)
            self.net.load_dict(state_dictsm)
            self.net.train()

            self.iou_predictor = self.net.bb_regressor

        self.layer_stride = {
            'conv0': 2,
            'conv1': 2,
            'block0': 4,
            'block1': 8,
            'block2': 16,
            'block3': 32,
            'classification': 16,
            'fc': None
        }
        self.layer_dim = {
            'conv0': 64,
            'conv1': 64,
            'block0': 256,
            'block1': 512,
            'block2': 1024,
            'block3': 2048,
            'classification': 256,
            'fc': None
        }

        self.iounet_feature_layers = self.net.bb_regressor_layer

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1] * len(self.output_layers)

        self.feature_layers = sorted(
            list(set(self.output_layers + self.iounet_feature_layers)))

        self.mean = np.reshape([0.485, 0.456, 0.406], [1, -1, 1, 1])
        self.std = np.reshape([0.229, 0.224, 0.225], [1, -1, 1, 1])

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net
        if hasattr(self, 'iou_predictor'):
            del self.iou_predictor
        if hasattr(self, 'iounet_backbone_features'):
            del self.iounet_backbone_features
        if hasattr(self, 'iounet_features'):
            del self.iounet_features

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([
            s * self.layer_stride[l]
            for l, s in zip(self.output_layers, self.pool_stride)
        ])

    def extract(self, im: np.ndarray, debug_save_name=None):
        with fluid.dygraph.guard():
            if debug_save_name is not None:
                np.savez(debug_save_name, im)

            im = im / 255.  # don't use im /= 255. since we don't want to alter the input
            im -= self.mean
            im /= self.std
            im = n2p(im)

            output_features = self.net.extract_features(im, self.feature_layers)

            # Store the raw resnet features which are input to iounet
            iounet_backbone_features = TensorList([
                output_features[layer] for layer in self.iounet_feature_layers
            ])
            self.iounet_backbone_features = iounet_backbone_features.numpy()

            # Store the processed features from iounet, just before pooling
            self.iounet_features = TensorList([
                f.numpy()
                for f in self.iou_predictor.get_iou_feat(
                    iounet_backbone_features)
            ])

            output = TensorList([
                output_features[layer].numpy() for layer in self.output_layers
            ])
            return output


class SFCAlexnet(MultiFeatureBase):
    """Alexnet feature.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """

    def __init__(self,
                 output_layers=('conv5', ),
                 net_path='estimator',
                 use_gpu=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.output_layers = list(output_layers)
        self.use_gpu = use_gpu
        self.net_path = net_path

    def initialize(self):
        with fluid.dygraph.guard():
            if os.path.isabs(self.net_path):
                net_path_full = self.net_path
            else:
                net_path_full = os.path.join(env_settings().network_path,
                                             self.net_path)

            self.net = siamfc_alexnet(
                backbone_pretrained=False,
                backbone_is_test=True,
                estimator_is_test=True)

            state_dictsm, _ = fluid.load_dygraph(net_path_full)
            self.net.load_dict(state_dictsm)
            self.net.train()

            self.target_estimator = self.net.target_estimator

        self.layer_stride = {'conv5': 8}
        self.layer_dim = {'conv5': 256}

        self.estimator_feature_layers = self.net.target_estimator_layer

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1] * len(self.output_layers)

        self.feature_layers = sorted(
            list(set(self.output_layers + self.estimator_feature_layers)))

        self.mean = np.reshape([0., 0., 0.], [1, -1, 1, 1])
        self.std = np.reshape([1 / 255., 1 / 255., 1 / 255.], [1, -1, 1, 1])

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net
        if hasattr(self, 'target_estimator'):
            del self.target_estimator
        if hasattr(self, 'estimator_backbone_features'):
            del self.estimator_backbone_features

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([
            s * self.layer_stride[l]
            for l, s in zip(self.output_layers, self.pool_stride)
        ])

    def extract(self, im: np.ndarray, debug_save_name=None):
        with fluid.dygraph.guard():
            if debug_save_name is not None:
                np.savez(debug_save_name, im)

            im = im / 255.  # don't use im /= 255. since we don't want to alter the input
            im -= self.mean
            im /= self.std
            im = n2p(im)

            output_features = self.net.extract_features(im, self.feature_layers)

            # Store the raw backbone features which are input to estimator
            estimator_backbone_features = TensorList([
                output_features[layer]
                for layer in self.estimator_feature_layers
            ])
            self.estimator_backbone_features = estimator_backbone_features.numpy(
            )

            output = TensorList([
                output_features[layer].numpy() for layer in self.output_layers
            ])
            return output


class SRPNAlexNet(MultiFeatureBase):
    """Alexnet feature.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """

    def __init__(self,
                 net_path='estimator',
                 use_gpu=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_gpu = use_gpu
        self.net_path = net_path

    def initialize(self):
        with fluid.dygraph.guard():
            if os.path.isabs(self.net_path):
                net_path_full = self.net_path
            else:
                net_path_full = os.path.join(env_settings().network_path, self.net_path)

            self.net = SiamRPN_AlexNet(backbone_pretrained=False, is_test=True)

            state_dict, _ = fluid.load_dygraph(net_path_full)
            self.net.load_dict(state_dict)
            self.net.eval()

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net

    def extract(self, im: np.ndarray, debug_save_name=None):
        with fluid.dygraph.guard():
            if debug_save_name is not None:
                np.savez(debug_save_name, im)

            im = n2p(im)

            output_features = self.net.extract_backbone_features(im)

            # Store the raw backbone features which are input to estimator
            output = TensorList([layer.numpy() for layer in output_features])
            return output


class SMaskResNet50_base(MultiFeatureBase):
    """Resnet50-dilated feature.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """

    def __init__(self,
                 net_path='estimator',
                 use_gpu=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_gpu = use_gpu
        self.net_path = net_path

    def initialize(self):
        with fluid.dygraph.guard():
            if os.path.isabs(self.net_path):
                net_path_full = self.net_path
            else:
                net_path_full = os.path.join(env_settings().network_path, self.net_path)

            self.net = SiamMask_ResNet50_base(backbone_pretrained=False, is_test=True)

            state_dict, _ = fluid.load_dygraph(net_path_full)
            self.net.load_dict(state_dict)
            self.net.eval()

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net

    def extract(self, im: np.ndarray, debug_save_name=None):
        with fluid.dygraph.guard():
            if debug_save_name is not None:
                np.savez(debug_save_name, im)

            im = n2p(im)

            output_features = self.net.extract_backbone_features(im)

            # Store the raw backbone features which are input to estimator
            output = TensorList([layer.numpy() for layer in output_features])
            return output


class SMaskResNet50_sharp(MultiFeatureBase):
    """Resnet50-dilated feature.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """

    def __init__(self,
                 net_path='estimator',
                 use_gpu=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_gpu = use_gpu
        self.net_path = net_path

    def initialize(self):
        with fluid.dygraph.guard():
            if os.path.isabs(self.net_path):
                net_path_full = self.net_path
            else:
                net_path_full = os.path.join(env_settings().network_path, self.net_path)

            self.net = SiamMask_ResNet50_sharp(backbone_pretrained=False, is_test=True)

            state_dict, _ = fluid.load_dygraph(net_path_full)
            self.net.load_dict(state_dict)
            self.net.eval()

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net

    def extract(self, im: np.ndarray, debug_save_name=None):
        with fluid.dygraph.guard():
            if debug_save_name is not None:
                np.savez(debug_save_name, im)

            im = n2p(im)

            output_features = self.net.extract_backbone_features(im)

            # Store the raw backbone features which are input to estimator
            output = TensorList([layer.numpy() for layer in output_features])
            return output
