from typing import Any, Callable, List, Optional, Sequence

import paddle
import paddle.nn as nn


class ConvNormActivation(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            stride: int=1,
            padding: Optional[int]=None,
            groups: int=1,
            norm_layer: Optional[Callable[..., nn.Layer]]=nn.BatchNorm2D,
            activation_layer: Optional[Callable[..., nn.Layer]]=nn.ReLU,
            dilation: int=1,
            bias: Optional[bool]=None, ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias_attr=bias, )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Layer):
    def __init__(self,
                 input_channels: int,
                 squeeze_channels: int,
                 activation: Callable[..., nn.Layer]=nn.ReLU,
                 scale_activation: Callable[..., nn.Layer]=nn.Sigmoid,
                 skip_se_quant: bool=True) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2D(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()
        if skip_se_quant:
            self.fc1.skip_quant = True
            self.fc2.skip_quant = True

    def _scale(self, input: paddle.Tensor) -> paddle.Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        scale = self._scale(input)
        return scale * input
