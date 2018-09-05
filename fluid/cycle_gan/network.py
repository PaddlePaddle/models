from model import *
import paddle.fluid as fluid

class CycleGAN():
    def __init__(self, input_A, input_B, fake_pool_A, fake_pool_B, optimizer):
        self.optimizer = optimizer
        g_program = fluid.default_main_program().clone()

        with fluid.program_guard(g_program):
            self.fake_B, self.printed = build_generator_resnet_9blocks(input_A, name="g_A")
            self.fake_A, _ = build_generator_resnet_9blocks(input_B, name="g_B")
            self.cyc_A, _ = build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B, _ = build_generator_resnet_9blocks(self.fake_A, "g_A")
            self.infer_program = g_program.clone()
            tmp = fluid.layers.elementwise_sub(x=input_A, y=self.cyc_A)
            tmp = fluid.layers.abs(tmp)
            tmp1 = fluid.layers.elementwise_sub(x=input_B, y=self.cyc_B)
            tmp1 = fluid.layers.abs(tmp1)
            self.cyc_loss = (fluid.layers.reduce_mean(tmp) + fluid.layers.reduce_mean(tmp1)) * 10

        self.g_A_program = g_program.clone()
        self.g_B_program = g_program.clone()

        with fluid.program_guard(self.g_A_program):
            self.fake_rec_B = build_gen_discriminator(self.fake_B, "d_B")
            self.disc_loss_B = fluid.layers.reduce_mean(fluid.layers.square(self.fake_rec_B - 1))
            self.g_loss_A = fluid.layers.elementwise_add(self.cyc_loss, self.disc_loss_B)
            vars=[]
            for var in self.g_A_program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("g_A"):
                    vars.append(var.name)
            vars=[] # stop training for debug
            self.optimizer._name = "g_A"
            self.optimizer.minimize(self.g_loss_A, parameter_list=vars)

            

        with fluid.program_guard(self.g_B_program):
            self.fake_rec_A = build_gen_discriminator(self.fake_A, "d_A")
            disc_loss_A = fluid.layers.reduce_mean(fluid.layers.square(self.fake_rec_A - 1))
            self.g_loss_B = fluid.layers.elementwise_add(self.cyc_loss, disc_loss_A)
            self.optimizer._name = "g_B"
            vars=[]
            for var in self.g_A_program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("g_B"):
                    vars.append(var.name)
            vars=[] # stop training for debug
            self.optimizer.minimize(self.g_loss_B, parameter_list=vars)

        self.d_A_program = fluid.default_main_program().clone()
        self.d_B_program = fluid.default_main_program().clone()
        with fluid.program_guard(self.d_A_program):
            self.rec_A = build_gen_discriminator(input_A, "d_A")
            self.fake_pool_rec_A = build_gen_discriminator(fake_pool_A, "d_A")
            self.d_loss_A = (fluid.layers.square(self.fake_pool_rec_A) + fluid.layers.square(self.rec_A - 1))/2.0
            self.d_loss_A = fluid.layers.reduce_mean(self.d_loss_A)
            self.optimizer._name = "d_A"
            vars=[]
            for var in self.g_A_program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("d_A"):
                    vars.append(var.name)
            self.optimizer.minimize(self.d_loss_A, parameter_list=vars)

        with fluid.program_guard(self.d_B_program):
            self.rec_B = build_gen_discriminator(input_B, "d_B")
            self.fake_pool_rec_B = build_gen_discriminator(fake_pool_B, "d_B")
            self.d_loss_B = (fluid.layers.square(self.fake_pool_rec_B) + fluid.layers.square(self.rec_B - 1))/2.0
            self.d_loss_B = fluid.layers.reduce_mean(self.d_loss_B)
            self.optimizer._name = "d_B"
            vars=[]
            for var in self.g_A_program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("d_B"):
                    vars.append(var.name)
            self.optimizer.minimize(self.d_loss_B, parameter_list=vars)

