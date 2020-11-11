import os
import numpy as np
import paddle
import paddle.fluid as fluid
from net import ESMM
import args
import logging
import utils

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def set_zero(place):
    auc_states_names = [
        'auc_1.tmp_0', 'auc_0.tmp_0'
    ]
    for name in auc_states_names:
        param = fluid.global_scope().var(name).get_tensor()
        if param:
            param_array = np.zeros(param._get_dims()).astype("int64")
            param.set(param_array, place)

def run_infer(args,model_path,test_data_path,vocab_size):
    place = fluid.CPUPlace()
    esmm_model = ESMM()
    
    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()

    with fluid.framework.program_guard(test_program, startup_program):
        with fluid.unique_name.guard():
            inputs = esmm_model.input_data()
            avg_cost,auc_ctr,auc_ctcvr= esmm_model.net(inputs, vocab_size, args.embed_size)
            
            dataset, file_list = utils.get_dataset(inputs, test_data_path,args.batch_size,args.cpu_num)
            
            exe = fluid.Executor(place)
            fluid.load(fluid.default_main_program(),os.path.join(model_path, "checkpoint"), exe)
            
            set_zero(place)
            
            dataset.set_filelist(file_list)
            exe.infer_from_dataset(program=test_program,
                                       dataset=dataset,
                                       fetch_list=[auc_ctr,auc_ctcvr],
                                       fetch_info=["auc_ctr","auc_ctcvr"],
                                       print_period=20,
                                       debug=False)
                                       
if __name__ == "__main__":
    import paddle
    paddle.enable_static()
  
    args = args.parse_args()
    model_list = []
    for _, dir, _ in os.walk(args.model_dir):
        for model in dir:
            if "epoch" in model:
                path = os.path.join(args.model_dir, model)
                model_list.append(path)
                
    vocab_size =utils.get_vocab_size(args.vocab_path)  
    
    for model in model_list:
        logger.info("Test model {}".format(model))
        run_infer(args, model,args.test_data_path)
                
             