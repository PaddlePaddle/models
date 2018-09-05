
from layers import *
import paddle.fluid as fluid

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width


batch_size = 1
pool_size = 50
ngf = 32
ndf = 64


def build_resnet_block(inputres, dim, name="resnet"):
#    out_res = fluid.layers.pad(inputres, [0, 0, 0, 0, 1, 1, 1, 1])
    out_res = fluid.layers.pad2d(inputres, [1, 1, 1, 1], mode="REFLECT")
    out_res = conv2d(out_res, dim, 3, 1, 0.02, "VALID", name+"_c1")
#    out_res = fluid.layers.pad(out_res, [0, 0, 0, 0, 1, 1, 1, 1])
    out_res = fluid.layers.pad2d(out_res, [1, 1, 1, 1], mode="REFLECT")
    out_res = conv2d(out_res, dim, 3, 1, 0.02, "VALID", name+ "_c2", relu=False)
    
    return fluid.layers.relu(out_res + inputres)


def build_generator_resnet_6blocks(inputgen, name="generator"):
    f = 7
    ks = 3
    
#    pad_input = fluid.layers.pad(inputgen,[0, 0, 0, 0, ks, ks, ks, ks])
    pad_input = fluid.layers.pad2d(inputgen,[ks, ks, ks, ks], mode="REFLECT")
    o_c1 = conv2d(pad_input, ngf, f, 1, 0.02, name=name+"_c1")
    o_c2 = conv2d(o_c1, ngf*2, ks, 2, 0.02, "SAME", name+"_c2")
    o_c3 = conv2d(o_c2, ngf*4, ks, 2, 0.02, "SAME", name+"_c3")

    o_r1 = build_resnet_block(o_c3, ngf*4, name+"_r1")
    o_r2 = build_resnet_block(o_r1, ngf*4, name+"_r2")
    o_r3 = build_resnet_block(o_r2, ngf*4, name+"_r3")
    o_r4 = build_resnet_block(o_r3, ngf*4, name+"_r4")
    o_r5 = build_resnet_block(o_r4, ngf*4, name+"_r5")
    o_r6 = build_resnet_block(o_r5, ngf*4, name+"_r6")

    o_c4 = deconv2d(o_r6, [batch_size,64,64,ngf*2], ngf*2, ks, 2, 0.02,"SAME", name+"_c4")
    o_c5 = deconv2d(o_c4, [batch_size,128,128,ngf], ngf, ks, 2, 0.02,"SAME", name+"_c5")
#    o_c5_pad = fluid.layers.pad(o_c5, [0, 0, 0, 0, ks, ks, ks, ks])
    o_c5_pad = fluid.layers.pad(o_c5, [ks, ks, ks, ks], mode="REFLECT")
    o_c6 = conv2d(o_c5_pad, img_layer, f, 1, 0.02, "VALID", name+"_c6", relu=False)

    # Adding the tanh layer
    out_gen = fluid.layers.tanh(o_c6, name+"_t1")
    return out_gen

def build_generator_resnet_9blocks(inputgen, name="generator"):
    '''The shape of input should be equal to the shape of output.'''
    f = 7
    ks = 3
#    pad_input = fluid.layers.pad(inputgen,[0, 0, 0, 0, ks, ks, ks, ks])
    pad_input = fluid.layers.pad2d(inputgen,[ks, ks, ks, ks], mode="REFLECT")
    o_c1 = conv2d(pad_input, ngf, f, 1, 0.02, name=name+"_c1")
    o_c2 = conv2d(o_c1, ngf*2, ks, 2, 0.02, "SAME", name+"_c2")
    o_c3 = conv2d(o_c2, ngf*4, ks, 2, 0.02, "SAME", name+"_c3")
    o_r1 = build_resnet_block(o_c3, ngf*4, name+"_r1")
    o_r2 = build_resnet_block(o_r1, ngf*4, name+"_r2")
    o_r3 = build_resnet_block(o_r2, ngf*4, name+"_r3")
    o_r4 = build_resnet_block(o_r3, ngf*4, name+"_r4")
    o_r5 = build_resnet_block(o_r4, ngf*4, name+"_r5")
    o_r6 = build_resnet_block(o_r5, ngf*4, name+"_r6")
    o_r7 = build_resnet_block(o_r6, ngf*4, name+"_r7")
    o_r8 = build_resnet_block(o_r7, ngf*4, name+"_r8")
    o_r9 = build_resnet_block(o_r8, ngf*4, name+"_r9")
    o_c4 = deconv2d(o_r9, [batch_size,128,128,ngf*2], ngf*2, ks, 2, 0.02, "SAME", name+"_c4")
    o_c5 = deconv2d(o_c4, [batch_size,256,256,ngf], ngf, ks, 2, 0.02,"SAME", name+"_c5")
    o_c6 = conv2d(o_c5, img_layer, f, 1, 0.02, "SAME", name+"_c6", relu=False)

    # Adding the tanh layer
    out_gen = fluid.layers.tanh(o_c6,name+"_t1")
    return out_gen, inputgen


def build_gen_discriminator(inputdisc, name="discriminator"):
    f = 4
    o_c1 = conv2d(inputdisc, ndf, f, 2, 0.02, "SAME", name+"_c1", norm=False, relufactor=0.2)
    o_c2 = conv2d(o_c1, ndf*2, f, 2, 0.02, "SAME", name+"_c2", relufactor=0.2)
    o_c3 = conv2d(o_c2, ndf*4, f, 2, 0.02, "SAME", name+"_c3", relufactor=0.2)
    o_c4 = conv2d(o_c3, ndf*8, f, 1, 0.02, "SAME", name+"_c4",relufactor=0.2)
    o_c5 = conv2d(o_c4, 1, f, 1, 0.02, "SAME", name+"_c5", norm=False, relu=False)
    return o_c5


#def patch_discriminator(inputdisc, name="discriminator"):
#    f= 4
#    patch_input = tf.random_crop(inputdisc,[1,70,70,3])
#    o_c1 = general_conv2d(patch_input, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm="False", relufactor=0.2)
#    o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
#    o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
#    o_c4 = general_conv2d(o_c3, ndf*8, f, f, 2, 2, 0.02, "SAME", "c4", relufactor=0.2)
#    o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)
#
#    return o_c5
