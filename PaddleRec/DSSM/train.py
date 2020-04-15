import numpy as np
import os
import paddle.fluid as fluid
from net import DSSM
import paddle
import args

def train(args):
    dssm_model = DSSM()
    inputs = dssm_model.input_data(args.TRIGRAM_D)
    
    avg_cost = dssm_model.net(inputs,args.TRIGRAM_D,args.L1_N,args.L2_N,args.L3_N,args.Neg,args.batch_size)
    
    # 选择反向更新优化策略
    optimizer = fluid.optimizer.SGD(learning_rate=args.base_lr)
    optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # Build a random data set.
    sample_size = 100
    l_Qs = []
    pos_l_Ds = []
    
    
    for i in range(sample_size):
        l_Q = np.random.rand(args.batch_size, args.TRIGRAM_D)
        l_Qs.append(l_Q)
        
        l_D = np.random.rand(args.batch_size, args.TRIGRAM_D)
        pos_l_Ds.append(l_D)
    for epoch in range(args.epochs):
        for i in range(sample_size):
            loss_data = exe.run(fluid.default_main_program(),
                                 feed={
                                     "query": l_Qs[i].astype('float32').reshape(args.batch_size,args.TRIGRAM_D),
                                     "doc_pos": pos_l_Ds[i].astype('float32').reshape(args.batch_size,args.TRIGRAM_D)
                                 },
                                 return_numpy=True,
                                 fetch_list=[avg_cost.name])
            print("epoch:%d,loss:%.5f"%(epoch,float(np.array(loss_data))))
        ##保存模型
        model_dir = os.path.join(args.model_dir,'epoch_' + str(epoch + 1), "checkpoint")
        main_program = fluid.default_main_program()
        fluid.io.save(main_program,model_dir)
        
if __name__ == "__main__":
    args = args.parse_args()
    train(args)

        
        
        