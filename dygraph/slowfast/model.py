#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn.functional as F

from model_utils import get_conv_init, get_bn_param_attr, ResStage, ResNetBasicHead
from batchnorm_helper import get_norm


class ResNetBasicStem(paddle.nn.Layer):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(self,
                 dim_in,
                 dim_out,
                 kernel,
                 stride,
                 padding,
                 eps=1e-5,
                 norm_module=paddle.nn.BatchNorm3D):
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.norm_module = norm_module
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        fan = (dim_out) * (self.kernel[0] * self.kernel[1] * self.kernel[2])
        initializer_tmp = get_conv_init(fan)

        self._conv = paddle.nn.Conv3D(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self._bn = self.norm_module(
            num_features=dim_out,
            epsilon=self.eps,
            weight_attr=get_bn_param_attr(),
            bias_attr=get_bn_param_attr(bn_weight=0.0))

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        x = F.relu(x)

        x = F.max_pool3d(
            x=x,
            kernel_size=[1, 3, 3],
            stride=[1, 2, 2],
            padding=[0, 1, 1],
            data_format="NCDHW")
        return x


class VideoModelStem(paddle.nn.Layer):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for slow and fast pathways.
    """

    def __init__(self,
                 dim_in,
                 dim_out,
                 kernel,
                 stride,
                 padding,
                 eps=1e-5,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            eps (float): epsilon for batch norm.
        """
        super(VideoModelStem, self).__init__()

        assert (len({
            len(dim_in),
            len(dim_out),
            len(kernel),
            len(stride),
            len(padding),
        }) == 1), "Input pathway dimensions are not consistent."
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.norm_module = norm_module
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        for pathway in range(len(dim_in)):
            stem = ResNetBasicStem(dim_in[pathway], dim_out[pathway],
                                   self.kernel[pathway], self.stride[pathway],
                                   self.padding[pathway], self.eps,
                                   self.norm_module)
            self.add_sublayer("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)

        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(paddle.to_tensor(x[pathway]))

        return x


class FuseFastToSlow(paddle.nn.Layer):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(self,
                 dim_in,
                 fusion_conv_channel_ratio,
                 fusion_kernel,
                 alpha,
                 eps=1e-5,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
        """
        super(FuseFastToSlow, self).__init__()
        fan = (dim_in * fusion_conv_channel_ratio) * (fusion_kernel * 1 * 1)
        initializer_tmp = get_conv_init(fan)

        self._conv_f2s = paddle.nn.Conv3D(
            in_channels=dim_in,
            out_channels=dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self._bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            epsilon=eps,
            weight_attr=get_bn_param_attr(),
            bias_attr=get_bn_param_attr(bn_weight=0.0))

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self._conv_f2s(x_f)
        fuse = self._bn(fuse)
        fuse = F.relu(fuse)
        x_s_fuse = paddle.concat(x=[x_s, fuse], axis=1, name=None)

        return [x_s_fuse, x_f]


class SlowFast(paddle.nn.Layer):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.DATA.num_classes
        self.num_frames = cfg.DATA.num_frames  #32
        self.alpha = cfg.DATA.alpha  #8
        self.beta = cfg.DATA.beta  #8
        self.crop_size = cfg.DATA.train_crop_size  #use train_crop_size even in test or infer
        self.norm_module = get_norm(cfg)
        self.num_pathways = 2
        self.res_depth = 50
        self.num_groups = 1
        self.input_channel_num = [3, 3]
        self.width_per_group = 64
        self.fusion_conv_channel_ratio = 2
        self.fusion_kernel_sz = 7  #TODO NOTE: modify to 7, not 5
        self.dropout_rate = 0.5
        self._construct_network()

    def _construct_network(self):
        """
        Builds a SlowFast model.
        The first pathway is the Slow pathway
        and the second pathway is the Fast pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        temp_kernel = [
            [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
            [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
            [[3], [3]],
        ]  # res5 temporal kernel for slow and fast pathway.

        self.s1 = VideoModelStem(
            dim_in=self.input_channel_num,
            dim_out=[self.width_per_group, self.width_per_group // self.beta],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module)
        self.s1_fuse = FuseFastToSlow(
            dim_in=self.width_per_group // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module)

        # ResNet backbone
        MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3)}
        (d2, d3, d4, d5) = MODEL_STAGE_DEPTH[self.res_depth]

        num_block_temp_kernel = [[3, 3], [4, 4], [6, 6], [3, 3]]
        spatial_dilations = [[1, 1], [1, 1], [1, 1], [1, 1]]
        spatial_strides = [[1, 1], [2, 2], [2, 2], [2, 2]]

        out_dim_ratio = self.beta // self.fusion_conv_channel_ratio  #4
        dim_inner = self.width_per_group * self.num_groups  #64

        self.s2 = ResStage(
            dim_in=[
                self.width_per_group + self.width_per_group // out_dim_ratio,
                self.width_per_group // self.beta,
            ],
            dim_out=[
                self.width_per_group * 4,
                self.width_per_group * 4 // self.beta,
            ],
            dim_inner=[dim_inner, dim_inner // self.beta],
            temp_kernel_sizes=temp_kernel[1],
            stride=spatial_strides[0],
            num_blocks=[d2] * 2,
            num_groups=[self.num_groups] * 2,
            num_block_temp_kernel=num_block_temp_kernel[0],
            dilation=spatial_dilations[0],
            norm_module=self.norm_module)

        self.s2_fuse = FuseFastToSlow(
            dim_in=self.width_per_group * 4 // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module, )

        self.s3 = ResStage(
            dim_in=[
                self.width_per_group * 4 + self.width_per_group * 4 //
                out_dim_ratio,
                self.width_per_group * 4 // self.beta,
            ],
            dim_out=[
                self.width_per_group * 8,
                self.width_per_group * 8 // self.beta,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // self.beta],
            temp_kernel_sizes=temp_kernel[2],
            stride=spatial_strides[1],
            num_blocks=[d3] * 2,
            num_groups=[self.num_groups] * 2,
            num_block_temp_kernel=num_block_temp_kernel[1],
            dilation=spatial_dilations[1],
            norm_module=self.norm_module, )

        self.s3_fuse = FuseFastToSlow(
            dim_in=self.width_per_group * 8 // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module, )

        self.s4 = ResStage(
            dim_in=[
                self.width_per_group * 8 + self.width_per_group * 8 //
                out_dim_ratio,
                self.width_per_group * 8 // self.beta,
            ],
            dim_out=[
                self.width_per_group * 16,
                self.width_per_group * 16 // self.beta,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // self.beta],
            temp_kernel_sizes=temp_kernel[3],
            stride=spatial_strides[2],
            num_blocks=[d4] * 2,
            num_groups=[self.num_groups] * 2,
            num_block_temp_kernel=num_block_temp_kernel[2],
            dilation=spatial_dilations[2],
            norm_module=self.norm_module, )

        self.s4_fuse = FuseFastToSlow(
            dim_in=self.width_per_group * 16 // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module, )

        self.s5 = ResStage(
            dim_in=[
                self.width_per_group * 16 + self.width_per_group * 16 //
                out_dim_ratio,
                self.width_per_group * 16 // self.beta,
            ],
            dim_out=[
                self.width_per_group * 32,
                self.width_per_group * 32 // self.beta,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // self.beta],
            temp_kernel_sizes=temp_kernel[4],
            stride=spatial_strides[3],
            num_blocks=[d5] * 2,
            num_groups=[self.num_groups] * 2,
            num_block_temp_kernel=num_block_temp_kernel[3],
            dilation=spatial_dilations[3],
            norm_module=self.norm_module, )

        self.pool_size = [[1, 1, 1], [1, 1, 1]]
        self.head = ResNetBasicHead(
            dim_in=[
                self.width_per_group * 32,
                self.width_per_group * 32 // self.beta,
            ],
            num_classes=self.num_classes,
            pool_size=[None, None] if self.cfg.MULTIGRID.SHORT_CYCLE else [
                [
                    self.num_frames // self.alpha // self.pool_size[0][0],
                    self.crop_size // 32 // self.pool_size[0][1],
                    self.crop_size // 32 // self.pool_size[0][2],
                ],
                [
                    self.num_frames // self.pool_size[1][0],
                    self.crop_size // 32 // self.pool_size[1][1],
                    self.crop_size // 32 // self.pool_size[1][2],
                ],
            ],
            dropout_rate=self.dropout_rate, )

    def forward(self, x):
        x = self.s1(x)  #VideoModelStem
        x = self.s1_fuse(x)  #FuseFastToSlow
        x = self.s2(x)  #ResStage
        x = self.s2_fuse(x)

        for pathway in range(self.num_pathways):
            x[pathway] = F.max_pool3d(
                x=x[pathway],
                kernel_size=self.pool_size[pathway],
                stride=self.pool_size[pathway],
                padding=[0, 0, 0],
                data_format="NCDHW")

        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.head(x)  #ResNetBasicHead
        return x
