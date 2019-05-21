from __future__ import print_function

from args import parse_args
import os
import paddle.fluid as fluid
import sys
from network_confv6 import ctr_deepfm_dataset


NUM_CONTEXT_FEATURE = 22
DIM_USER_PROFILE = 10
DIM_DENSE_FEATURE = 3
PYTHON_PATH = "/home/yaoxuefeng/whls/paddle_release_home/python/bin/python" # this is mine change yours

def train():
    args = parse_args()
    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)
    
    #set the input format for our model. Note that you need to carefully modify them when you define a new network
    #user_profile = fluid.layers.data(
        #name="user_profile", shape=[DIM_USER_PROFILE], dtype='int64', lod_level=1)
    dense_feature = fluid.layers.data(
        name="dense_feature", shape=[DIM_DENSE_FEATURE], dtype='float32')
    context_feature = [
        fluid.layers.data(name="context" + str(i), shape=[1], lod_level=1, dtype="int64")
        for i in range(0, NUM_CONTEXT_FEATURE)]
    context_feature_fm = fluid.layers.data(
        name="context_fm", shape=[1], dtype='int64', lod_level=1)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    print("ready to network")
    #self define network 
    loss, auc_var, batch_auc_var, accuracy, predict = ctr_deepfm_dataset(dense_feature, context_feature, context_feature_fm, label,
                                                        args.embedding_size, args.sparse_feature_dim)

    print("ready to optimize")
    optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
    optimizer.minimize(loss)
    #single machine CPU training. more options on trainig please visit PaddlePaddle site
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    #use dataset api for much faster speed
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var([dense_feature] + context_feature + [context_feature_fm] + [label])
    #self define how to process generated training insatnces in map_reader.py
    pipe_command = PYTHON_PATH + "  map_reader.py %d" % args.sparse_feature_dim
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(args.batch_size)
    thread_num = 1
    dataset.set_thread(thread_num)
    #self define how to split training files for example:"split -a 2 -d -l 200000 normed_train.txt normed_train"
    whole_filelist = ["./out/normed_train%d" % x for x in range(len(os.listdir("out")))]
    whole_filelist = ["./out/normed_train00", "./out/normed_train01", "./out/normed_train02", "./out/normed_train03",
                      "./out/normed_train04", "./out/normed_train05", "./out/normed_train06", "./out/normed_train07",
                      "./out/normed_train08",
                      "./out/normed_train09", "./out/normed_train10", "./out/normed_train11"]
    print("ready to epochs")
    epochs = 10
    for i in range(epochs):
        print("start %dth epoch" % i)
        dataset.set_filelist(whole_filelist[:int(len(whole_filelist))])
        #print the informations you want by setting fetch_list and fetch_info
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset,
                               fetch_list=[auc_var, accuracy, predict, label],
                               fetch_info=["auc", "accuracy", "predict", "label"],
                               debug=False)
        model_dir = args.model_output_dir + '/epoch' + str(i + 1) + ".model"
        sys.stderr.write("epoch%d finished" % (i + 1))
        #save model
        fluid.io.save_inference_model(model_dir, [dense_feature.name] + [x.name for x in context_feature] + [context_feature_fm.name] + [label.name],
                                      [loss, auc_var, accuracy, predict, label], exe)


if __name__ == '__main__':
    train()
