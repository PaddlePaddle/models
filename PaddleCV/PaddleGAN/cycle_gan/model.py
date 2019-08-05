from layers import conv2d, deconv2d
import paddle.fluid as fluid


def build_resnet_block(inputres, dim, name="resnet"):
    out_res = fluid.layers.pad2d(inputres, [1, 1, 1, 1], mode="reflect")
    out_res = conv2d(out_res, dim, 3, 1, 0.02, "VALID", name + "_c1")
    out_res = fluid.layers.pad2d(out_res, [1, 1, 1, 1], mode="reflect")
    out_res = conv2d(
        out_res, dim, 3, 1, 0.02, "VALID", name + "_c2", relu=False)
    return fluid.layers.relu(out_res + inputres)


def build_generator_resnet_9blocks(inputgen, name="generator"):
    '''The shape of input should be equal to the shape of output.'''
    pad_input = fluid.layers.pad2d(inputgen, [3, 3, 3, 3], mode="reflect")
    o_c1 = conv2d(pad_input, 32, 7, 1, 0.02, name=name + "_c1")
    o_c2 = conv2d(o_c1, 64, 3, 2, 0.02, "SAME", name + "_c2")
    o_c3 = conv2d(o_c2, 128, 3, 2, 0.02, "SAME", name + "_c3")
    o_r1 = build_resnet_block(o_c3, 128, name + "_r1")
    o_r2 = build_resnet_block(o_r1, 128, name + "_r2")
    o_r3 = build_resnet_block(o_r2, 128, name + "_r3")
    o_r4 = build_resnet_block(o_r3, 128, name + "_r4")
    o_r5 = build_resnet_block(o_r4, 128, name + "_r5")
    o_r6 = build_resnet_block(o_r5, 128, name + "_r6")
    o_r7 = build_resnet_block(o_r6, 128, name + "_r7")
    o_r8 = build_resnet_block(o_r7, 128, name + "_r8")
    o_r9 = build_resnet_block(o_r8, 128, name + "_r9")
    o_c4 = deconv2d(o_r9, [128, 128], 64, 3, 2, 0.02, "SAME", name + "_c4")
    o_c5 = deconv2d(o_c4, [256, 256], 32, 3, 2, 0.02, "SAME", name + "_c5")
    o_c6 = conv2d(o_c5, 3, 7, 1, 0.02, "SAME", name + "_c6", relu=False)

    out_gen = fluid.layers.tanh(o_c6, name + "_t1")
    return out_gen


def build_gen_discriminator(inputdisc, name="discriminator"):
    o_c1 = conv2d(
        inputdisc,
        64,
        4,
        2,
        0.02,
        "SAME",
        name + "_c1",
        norm=False,
        relufactor=0.2)
    o_c2 = conv2d(o_c1, 128, 4, 2, 0.02, "SAME", name + "_c2", relufactor=0.2)
    o_c3 = conv2d(o_c2, 256, 4, 2, 0.02, "SAME", name + "_c3", relufactor=0.2)
    o_c4 = conv2d(o_c3, 512, 4, 1, 0.02, "SAME", name + "_c4", relufactor=0.2)
    o_c5 = conv2d(
        o_c4, 1, 4, 1, 0.02, "SAME", name + "_c5", norm=False, relu=False)
    return o_c5
