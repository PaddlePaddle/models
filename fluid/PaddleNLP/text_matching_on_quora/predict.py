"""
usage:
    python predict.py  \
        --model=model_name\
        --config= \
        --epoch=epoch_id
    This model will evaluate the model on dataset defined in config; The trained checkpoint to be load is config.model_dir/epoch_id
"""

import os, sys
import argparse
import paddle
import paddle.fluid as fluid
import utils, configs, models
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('--model_name',       type=str,   default='cdssmNet',                  help="Which model to train")
parser.add_argument('--config',           type=str,   default='cdssm_base',       help="The global config setting")
parser.add_argument('--epoch',            type=int,   default=30,       help="The checkpoint to be load for predict")

def predict(reader, net, config):
    """
    predicte function
    """ 
    if config.save_dirname is None:
        print(str(model_path) + " cannot be found")
        return

    if config.use_cuda:
        print("Using GPU")
        place = fluid.CUDAPlace(0)
    else:
        print("Using CPU")
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    res_file = open('QQP.tsv', 'w')
    res_file.write('index' + '\t' + 'prediction' + '\n')
    idx = 0
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                os.path.join(config.save_dirname + '_for_predict', 'epoch' + str(config.epoch_id)), exe)
        feed_list = [inference_program.global_block().var(x) for x in feed_target_names]
        feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        all = 0
        for data in reader():
            pred, = exe.run(inference_program,
                          feed=feeder.feed(data),
                          fetch_list=fetch_targets,
                          return_numpy=True)
            pred = pred.tolist()
            pred = np.argmax(pred, axis=1)
            for p in pred:
                res_file.write(str(idx) + '\t' + str(p) + '\n')
                idx += 1
            all += len(data)
        print(all)
    
def main():
    args = parser.parse_args()
    global_config = configs.__dict__[args.config]()
    global_config.epoch_id = args.epoch
    sys.stderr.write("net_name: %s\n" % args.model_name)
    net = models.__dict__[args.model_name](global_config)

    # get word_dict
    word_dict = utils.getDict(data_type=global_config.data_type)
    
    reader = utils.prepare_test_data(
                 data_type=global_config.data_type,
                 word_dict=word_dict,
                 batch_size = global_config.batch_size,
                 buf_size=800000,
                 use_pad=(not global_config.use_lod_tensor))

    # use cuda or not
    if not global_config.has_member('use_cuda'):
        if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] != '':
            global_config.use_cuda = True
        else:
            global_config.use_cuda = False

    global_config.list_config()
    predict(reader, net, global_config)

if __name__ ==  '__main__':
    main()
