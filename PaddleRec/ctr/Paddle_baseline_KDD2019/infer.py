import argparse
import logging

import numpy as np
# disable gpu training for this example
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import paddle
import paddle.fluid as fluid

import map_reader
from network_conf import ctr_deepfm_dataset

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle DeepFM example")
    parser.add_argument(
        '--model_path',
        type=str,
        #required=True,
        default='models',
        help="The path of model parameters gz file")
    parser.add_argument(
        '--data_path',
        type=str,
        required=False,
        help="The path of the dataset to infer")
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=16,
        help="The size for embedding layer (default:10)")
    parser.add_argument(
        '--sparse_feature_dim',
        type=int,
        default=1000001,
        help="The size for embedding layer (default:1000001)")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")

    return parser.parse_args()


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def data2tensor(data, place):
    feed_dict = {}
    test_dict = {}
    dense = data[0]
    sparse = data[1:-1]
    y = data[-1]
    dense_data = np.array([x[0] for x in data]).astype("float32")
    dense_data = dense_data.reshape([-1, 65])
    feed_dict["user_profile"] = dense_data
    for i in range(10):
        sparse_data = to_lodtensor([x[1 + i] for x in data], place)
        feed_dict["context" + str(i)] = sparse_data

    y_data = np.array([x[-1] for x in data]).astype("int64")
    y_data = y_data.reshape([-1, 1])
    feed_dict["label"] = y_data
    test_dict["test"] = [1]
    return feed_dict, test_dict


def infer():
    args = parse_args()

    place = fluid.CPUPlace()
    inference_scope = fluid.core.Scope()

    filelist = ["%s/%s" % (args.data_path, x) for x in os.listdir(args.data_path)]
    from map_reader import MapDataset
    map_dataset = MapDataset()
    map_dataset.setup(args.sparse_feature_dim)
    exe = fluid.Executor(place)

    whole_filelist = ["raw_data/part-%d" % x for x in range(len(os.listdir("raw_data")))]
    #whole_filelist = ["./out/normed_train09",  "./out/normed_train10",  "./out/normed_train11"]
    test_files = whole_filelist[int(0.0 * len(whole_filelist)):int(1.0 * len(whole_filelist))]

    # file_groups = [whole_filelist[i:i+train_thread_num] for i in range(0, len(whole_filelist), train_thread_num)]

    def set_zero(var_name):
        param = inference_scope.var(var_name).get_tensor()
        param_array = np.zeros(param._get_dims()).astype("int64")
        param.set(param_array, place)

    epochs = 2
    for i in range(epochs):
        cur_model_path = args.model_path + "/epoch" + str(i + 1) + ".model"
        with fluid.scope_guard(inference_scope):
            [inference_program, feed_target_names, fetch_targets] = \
                fluid.io.load_inference_model(cur_model_path, exe)
            auc_states_names = ['_generated_var_2', '_generated_var_3']
            for name in auc_states_names:
                set_zero(name)

            test_reader = map_dataset.infer_reader(test_files, 1000, 100000)
            for batch_id, data in enumerate(test_reader()):
                loss_val, auc_val, accuracy, predict, label = exe.run(inference_program,
                                            feed=data2tensor(data, place),
                                            fetch_list=fetch_targets, return_numpy=False)

                #print(np.array(predict))
                #x = np.array(predict)
                #print(.shape)x
            #print("train_pass_%d, test_pass_%d\t%f\t" % (i - 1, i, auc_val))


if __name__ == '__main__':
    infer()
