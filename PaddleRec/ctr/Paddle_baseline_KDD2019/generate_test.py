# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
import numpy as np
# disable gpu training for this example
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import paddle
import paddle.fluid as fluid
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)
num_context_feature = 22

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
    dense = data[0]
    sparse = data[1:-1]
    y = data[-1]
    #user_data = np.array([x[0] for x in data]).astype("float32")
    #user_data = user_data.reshape([-1, 10])
    #feed_dict["user_profile"] = user_data
    dense_data = np.array([x[0] for x in data]).astype("float32")
    dense_data = dense_data.reshape([-1, 3])
    feed_dict["dense_feature"] = dense_data
    for i in range(num_context_feature):
        sparse_data = to_lodtensor([x[1 + i] for x in data], place)
        feed_dict["context" + str(i)] = sparse_data

    context_fm = to_lodtensor(np.array([x[-2] for x in data]).astype("float32"), place)

    feed_dict["context_fm"] = context_fm
    y_data = np.array([x[-1] for x in data]).astype("int64")
    y_data = y_data.reshape([-1, 1])
    feed_dict["label"] = y_data
    return feed_dict

def test():
    args = parse_args()

    place = fluid.CPUPlace()
    test_scope = fluid.core.Scope()

    # filelist = ["%s/%s" % (args.data_path, x) for x in os.listdir(args.data_path)]
    from map_reader import MapDataset
    map_dataset = MapDataset()
    map_dataset.setup(args.sparse_feature_dim)
    exe = fluid.Executor(place)

    whole_filelist = ["./out/normed_test_session.txt"]
    test_files = whole_filelist[int(0.0 * len(whole_filelist)):int(1.0 * len(whole_filelist))]


    epochs = 1

    for i in range(epochs):
        cur_model_path = args.model_path + "/epoch" + str(1) + ".model"
        with open("./testres/res" + str(i), 'w') as r:
            with fluid.scope_guard(test_scope):
                [inference_program, feed_target_names, fetch_targets] = \
                    fluid.io.load_inference_model(cur_model_path, exe)

                test_reader = map_dataset.test_reader(test_files, 1000, 100000)
                k = 0
                for batch_id, data in enumerate(test_reader()):
                    print(len(data[0]))
                    feed_dict = data2tensor(data, place)
                    loss_val, auc_val, accuracy, predict, _ = exe.run(inference_program,
                                                feed=feed_dict,
                                                fetch_list=fetch_targets, return_numpy=False)

                    x = np.array(predict)
                    for j in range(x.shape[0]):
                        r.write(str(x[j][1]))
                        r.write("\n")


if __name__ == '__main__':
    test()
