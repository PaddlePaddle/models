import paddle
import paddle.fluid as fluid

class Measurement(object):
    def  __init__(self,image,model,args,label): 
        self.class_num = args.class_num
        self.model_name = args.model_name
        self.use_mixup = args.use_mixup
        self.use_label_smoothing = args.use_label_smoothing
        self.epsilon = args.epsilon
        self.image = image
        self.model = model
        self.label = label
    
    def out(self):
        net_out = self.model.net(input=self.image, class_dim=self.class_dim)
        softmax_out = fluid.layers.softmax(net_out, use_cudnn=False)
        cost = fluid.layers.cross_entropy(input=softmax_out, label=label)
        avg_cost =  fluid.layers.mean(self.cost)
        acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=softmax_out, label=label, k=5)

        return [avg_cost, avg_top1, avg_top5]

    def out_label_smoothing(self):
        net_out = self.model.net(input=self.image, class_dim=self.class_dim)
        softmax_out = fluid.layers.softmax(net_out, use_cudnn=False)
        label_one_hot = fluid.layers.one_hot(input=self.label, depth=self.class_dim)
        smooth_label = fluid.layers.label_smooth(label=label_one_hot, epsilon=self.epsilon, dtype="float32")

        cost = fluid.layers.cross_entropy(input=softmax_out, label=smooth_label, soft_label=True)

        avg_cost =  fluid.layers.mean(self.cost)
        acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=softmax_out, label=label, k=5)

        return [avg_cost, avg_top1, avg_top5]


class GoogleNet_Measurement(Measurement):
    def __init__(self):
        super(GoogleNet_Measurement,self).__init__()
    def out(self):
        out0, out1, out2 = self.model.net(input=self.image, class_dim=self.class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
        return [avg_cost, avg_top1, avg_top5]

class Distill_Measurement(Measurement):
    def __init__(self):
        super(Distill_Measurement,self).__init__()
    def out(self):
        out1, out2 = self.model.net(input=self.image, class_dim=self.class_dim)
        softmax_out1, softmax_out = fluid.layers.softmax(out1), fluid.layers.softmax(out2)
        smooth_out1 = fluid.layers.label_smooth(label=softmax_out1, epsilon=0.0, dtype="float32")
        cost = fluid.layers.cross_entropy(input=softmax_out, label=smooth_out1, soft_label=True)
        avg_cost = fluid.layers.mean(cost)

        return avg_cost

class Mixup_Measuremeat(Measurement):
    def __init__(self):
        super(Mixup_Measuremeat, self).__init__()

    def out(self):

        loss_a = calc_loss(epsilon,y_a,class_dim,softmax_out,use_label_smoothing)
        loss_b = calc_loss(epsilon,y_b,class_dim,softmax_out,use_label_smoothing)
        loss_a_mean = fluid.layers.mean(x = loss_a)
        loss_b_mean = fluid.layers.mean(x = loss_b)
        cost = lam * loss_a_mean + (1 - lam) * loss_b_mean
        avg_cost = fluid.layers.mean(x=cost)
        if args.scale_loss > 1:
            avg_cost = fluid.layers.mean(x=cost) * float(args.scale_loss)
        return avg_cost
       
