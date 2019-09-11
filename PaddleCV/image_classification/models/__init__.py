#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from .alexnet import AlexNet
from .mobilenet_v1 import MobileNetV1_x0_25, MobileNetV1_x0_5, MobileNetV1_x1_0, MobileNetV1_x0_75, MobileNetV1
from .mobilenet_v2 import MobileNetV2_x0_25, MobileNetV2_x0_5, MobileNetV2_x0_75, MobileNetV2_x1_0, MobileNetV2_x1_5, MobileNetV2_x2_0, MobileNetV2
from .mobilenet_v3 import MobileNetV3_small_x0_25, MobileNetV3_small_x0_5, MobileNetV3_small_x0_75, MobileNetV3_small_x1_0, MobileNetV3_small_x1_25, MobileNetV3_large_x0_25, MobileNetV3_large_x0_5, MobileNetV3_large_x0_75, MobileNetV3_large_x1_0, MobileNetV3_large_x1_25
from .googlenet import GoogLeNet
from .vgg import VGG11, VGG13, VGG16, VGG19
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .resnet_vc import ResNet50_vc, ResNet101_vc, ResNet152_vc
from .resnet_vd import ResNet18_vd, ResNet34_vd, ResNet50_vd, ResNet101_vd, ResNet152_vd, ResNet200_vd
from .resnext import ResNeXt50_64x4d, ResNeXt101_64x4d, ResNeXt152_64x4d, ResNeXt50_32x4d, ResNeXt101_32x4d, ResNeXt152_32x4d
from .resnext_vd import ResNeXt50_vd_64x4d, ResNeXt101_vd_64x4d, ResNeXt152_vd_64x4d, ResNeXt50_vd_32x4d, ResNeXt101_vd_32x4d, ResNeXt152_vd_32x4d
from .inception_v4 import InceptionV4
from .se_resnet_vd import SE_ResNet18_vd, SE_ResNet34_vd, SE_ResNet50_vd, SE_ResNet101_vd, SE_ResNet152_vd, SE_ResNet200_vd
from .se_resnext import SE_ResNeXt50_32x4d, SE_ResNeXt101_32x4d, SE_ResNeXt152_32x4d
from .se_resnext_vd import SE_ResNeXt50_vd_32x4d, SE_ResNeXt101_vd_32x4d, SENet154_vd
from .dpn import DPN68, DPN92, DPN98, DPN107, DPN131
from .shufflenet_v2_swish import ShuffleNetV2_swish, ShuffleNetV2_x0_5_swish, ShuffleNetV2_x1_0_swish, ShuffleNetV2_x1_5_swish, ShuffleNetV2_x2_0_swish
from .shufflenet_v2 import ShuffleNetV2_x0_25, ShuffleNetV2_x0_33, ShuffleNetV2_x0_5, ShuffleNetV2_x1_0, ShuffleNetV2_x1_5, ShuffleNetV2_x2_0, ShuffleNetV2
from .fast_imagenet import FastImageNet
from .xception import Xception41, Xception65, Xception71
from .xception_deeplab import Xception41_deeplab, Xception65_deeplab, Xception71_deeplab
from .densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201, DenseNet264
from .squeezenet import SqueezeNet1_0, SqueezeNet1_1
from .darknet import DarkNet53
from .resnext101_wsl import ResNeXt101_32x8d_wsl, ResNeXt101_32x16d_wsl, ResNeXt101_32x32d_wsl, ResNeXt101_32x48d_wsl, Fix_ResNeXt101_32x48d_wsl
