import paddle.fluid as fluid
import numpy as np
import sys
import args
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def infer(args):
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    with fluid.scope_guard(fluid.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(args.model_dir, exe)
        
        #构造测试数据
        sample_size = 100
        l_Qs = []
        pos_l_Ds = []
        
        for i in range(sample_size):
            l_Q = np.random.rand(1, args.TRIGRAM_D)
            l_Qs.append(l_Q)
            
            l_D = np.random.rand(1, args.TRIGRAM_D)
            pos_l_Ds.append(l_D)
            
        res = []
        for i in range(sample_size):
            con_sim = exe.run(infer_program,
                   feed={"query": l_Qs[i].astype('float32').reshape(1,args.TRIGRAM_D),
                         "doc_pos": pos_l_Ds[i].astype('float32').reshape(1,args.TRIGRAM_D)},
                   fetch_list=fetch_vars,
                   return_numpy=True)

            logger.info("query_doc_sim: {:.5f}".format(np.array(con_sim).reshape(-1,1)[0][0]))

if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = args.parse_args()
    infer(args)