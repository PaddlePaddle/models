import paddle.fluid as fluid
from forwardAttentionLayer import ForwardAttention
from reverseAttentionLayer import ReverseAttention, ReverseMaskConv


class LBAMModel():
    def __init__(self, num_filters):
        self.num_filters = num_filters

    def net(self, inputImgs, masks):
        ef1, mu1, skipConnect1, forwardMap1 = ForwardAttention(
            inputImgs, masks, 64, bn=False)
        ef2, mu2, skipConnect2, forwardMap2 = ForwardAttention(ef1, mu1, 128)
        ef3, mu3, skipConnect3, forwardMap3 = ForwardAttention(ef2, mu2, 256)
        ef4, mu4, skipConnect4, forwardMap4 = ForwardAttention(ef3, mu3, 512)

        ef5, mu5, skipConnect5, forwardMap5 = ForwardAttention(ef4, mu4, 512)
        ef6, mu6, skipConnect6, forwardMap6 = ForwardAttention(ef5, mu5, 512)
        ef7, _, _, _ = ForwardAttention(ef6, mu6, 512)

        reverseMap1, revMu1 = ReverseMaskConv(1 - masks, 64)
        reverseMap2, revMu2 = ReverseMaskConv(revMu1, 128)
        reverseMap3, revMu3 = ReverseMaskConv(revMu2, 256)
        reverseMap4, revMu4 = ReverseMaskConv(revMu3, 512)
        reverseMap5, revMu5 = ReverseMaskConv(revMu4, 512)
        reverseMap6, _ = ReverseMaskConv(revMu5, 512)

        concatMap6 = fluid.layers.concat([forwardMap6, reverseMap6], axis=1)
        dcFeatures1 = ReverseAttention(skipConnect6, ef7, concatMap6, 512)

        concatMap5 = fluid.layers.concat([forwardMap5, reverseMap5], axis=1)
        dcFeatures2 = ReverseAttention(skipConnect5, dcFeatures1, concatMap5,
                                       512)

        concatMap4 = fluid.layers.concat([forwardMap4, reverseMap4], axis=1)
        dcFeatures3 = ReverseAttention(skipConnect4, dcFeatures2, concatMap4,
                                       512)

        concatMap3 = fluid.layers.concat([forwardMap3, reverseMap3], axis=1)
        dcFeatures4 = ReverseAttention(skipConnect3, dcFeatures3, concatMap3,
                                       256)

        concatMap2 = fluid.layers.concat([forwardMap2, reverseMap2], axis=1)
        dcFeatures5 = ReverseAttention(skipConnect2, dcFeatures4, concatMap2,
                                       128)

        concatMap1 = fluid.layers.concat([forwardMap1, reverseMap1], axis=1)
        dcFeatures6 = ReverseAttention(skipConnect1, dcFeatures5, concatMap1,
                                       64)

        dcFeatures7 = fluid.layers.conv2d_transpose(
            input=dcFeatures6,
            num_filters=self.num_filters,
            filter_size=4,
            stride=2,
            padding=1,
            act=None,
            bias_attr=False)

        output = fluid.layers.abs(fluid.layers.tanh(dcFeatures7))

        return output
