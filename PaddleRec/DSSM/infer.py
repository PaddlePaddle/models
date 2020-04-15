import os
import numpy as np
import paddle
import paddle.fluid as fluid
import args
from net import DSSM
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def run_infer(args,model_path):
    place = fluid.CPUPlace()
    dssm_model = DSSM()
    
    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()

    with fluid.framework.program_guard(test_program, startup_program):
        with fluid.unique_name.guard():
            inputs = dssm_model.input_data(args.TRIGRAM_D)
            avg_cost = dssm_model.net(inputs,args.TRIGRAM_D,args.L1_N,args.L2_N,args.L3_N,args.Neg,args.batch_size)
            
            exe = fluid.Executor(place)
            #加载模型
            fluid.load(fluid.default_main_program(),os.path.join(model_path, "checkpoint"), exe)
        
            #构造测试数据
            sample_size = 100
            l_Qs = []
            pos_l_Ds = []
            
            for i in range(sample_size):
                l_Q = np.random.rand(args.batch_size, args.TRIGRAM_D)
                l_Qs.append(l_Q)
                
                l_D = np.random.rand(args.batch_size, args.TRIGRAM_D)
                pos_l_Ds.append(l_D)
    
            for i in range(sample_size):
                loss_data = exe.run(test_program,
                       feed={"query": l_Qs[i].astype('float32').reshape(args.batch_size,args.TRIGRAM_D),
                             "doc_pos": pos_l_Ds[i].astype('float32').reshape(args.batch_size,args.TRIGRAM_D)},
                       fetch_list=[avg_cost.name],
                       return_numpy=True)
                print("loss:%.5f"%(float(np.array(loss_data))))
                                       
if __name__ == "__main__":
  
    args = args.parse_args()
    model_list = []
    
    for epoch in range(1, args.epochs):
        model_path = args.model_dir + "/epoch_" + str(epoch)
        run_infer(args, model_path)
         
                
                
                
                
                
                
                
                

    