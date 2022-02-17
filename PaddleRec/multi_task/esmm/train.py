import numpy as np
import os
import paddle.fluid as fluid
from net import ESMM
import paddle
import utils
import args

def train(args, vocab_size, train_data_path):
    esmm_model = ESMM()
    inputs = esmm_model.input_data()

    dataset, file_list = utils.get_dataset(inputs, train_data_path,args.batch_size,args.cpu_num)
    
    avg_cost,auc_ctr,auc_ctcvr= esmm_model.net(inputs, vocab_size, args.embed_size)
    optimizer = fluid.optimizer.Adam()
    optimizer.minimize(avg_cost)
    
    if args.use_gpu == True:
        exe = fluid.Executor(fluid.CUDAPlace(0))
        dataset.set_thread(1)
    else:
        exe = fluid.Executor(fluid.CPUPlace())
        dataset.set_thread(args.cpu_num)
    
    exe.run(fluid.default_startup_program())

    for epoch in range(args.epochs):
        dataset.set_filelist(file_list)
        exe.train_from_dataset(program=fluid.default_main_program(),
                                   dataset=dataset,
                                   fetch_list=[avg_cost,auc_ctr,auc_ctcvr],
                                   fetch_info=['epoch %d batch loss' % (epoch), "auc_ctr","auc_ctcvr"],
                                   print_period=20,
                                   debug=False)
        model_dir = os.path.join(args.model_dir,'epoch_' + str(epoch + 1), "checkpoint")
        main_program = fluid.default_main_program()
        fluid.io.save(main_program,model_dir)

if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = args.parse_args()
    vocab_size =utils.get_vocab_size(args.vocab_path)
    train(args, vocab_size, args.train_data_path)
