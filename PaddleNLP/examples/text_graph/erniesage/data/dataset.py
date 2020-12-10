# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np

import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import IterableDataset
import pgl
from pgl.utils.logger import log
from pgl.sample import alias_sample, graphsage_sample

__all__ = [
    "TrainData",
    "PredictData",
    "GraphDataset",
]


class TrainData(object):
    def __init__(self, graph_work_path):
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        print("trainer_id: %s, trainer_count: %s." %
              (trainer_id, trainer_count))

        edges = np.load(
            os.path.join(graph_work_path, "train_data.npy"), allow_pickle=True)
        # edges is bidirectional.
        train_src = edges[trainer_id::trainer_count, 0]
        train_dst = edges[trainer_id::trainer_count, 1]
        returns = {"train_data": [train_src, train_dst]}

        if os.path.exists(os.path.join(graph_work_path, "neg_samples.npy")):
            neg_samples = np.load(
                os.path.join(graph_work_path, "neg_samples.npy"),
                allow_pickle=True)
            if neg_samples.size != 0:
                train_negs = neg_samples[trainer_id::trainer_count]
                returns["train_data"].append(train_negs)
        print("Load train_data done.")
        self.data = returns

    def __getitem__(self, index):
        return [data[index] for data in self.data["train_data"]]

    def __len__(self):
        return len(self.data["train_data"][0])


class PredictData(object):
    def __init__(self, num_nodes):
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        self.data = np.arange(trainer_id, num_nodes, trainer_count)

    def __getitem__(self, index):
        return [self.data[index], self.data[index]]

    def __len__(self):
        return len(self.data)


class GraphDataset(IterableDataset):
    """load graph, sample, feed as numpy list.
    """

    def __init__(self,
                 graphs,
                 data,
                 batch_size,
                 samples,
                 mode,
                 graph_data_path,
                 shuffle=True,
                 neg_type="batch_neg"):
        """[summary]

        Args:
            graphs (GraphTensor List): GraphTensor of each layers.
            data (List): train/test source list. Can be edges or nodes.
            batch_size (int): the batch size means the edges num to be sampled.
            samples (List): List of sample number for each layer.
            mode (str): train, eval, test
            graph_data_path (str): the real graph object.
            shuffle (bool, optional): shuffle data. Defaults to True.
            neg_type (str, optional): negative sample methods. Defaults to "batch_neg".
        """

        super(GraphDataset, self).__init__()
        self.line_examples = data
        self.graphs = graphs
        self.samples = samples
        self.mode = mode
        self.load_graph(graph_data_path)
        self.num_layers = len(graphs)
        self.neg_type = neg_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = 1

    def load_graph(self, graph_data_path):
        self.graph = pgl.graph.MemmapGraph(graph_data_path)
        self.alias = np.load(
            os.path.join(graph_data_path, "alias.npy"), mmap_mode="r")
        self.events = np.load(
            os.path.join(graph_data_path, "events.npy"), mmap_mode="r")
        self.term_ids = np.load(
            os.path.join(graph_data_path, "term_ids.npy"), mmap_mode="r")

    def batch_fn(self, batch_ex):
        # batch_ex = [
        #     (src, dst, neg),
        #     (src, dst, neg),
        #     (src, dst, neg),
        #     ]
        batch_src = []
        batch_dst = []
        batch_neg = []
        for batch in batch_ex:
            batch_src.append(batch[0])
            batch_dst.append(batch[1])
            if len(batch) == 3:  # default neg samples
                batch_neg.append(batch[2])

        if len(batch_src) != self.batch_size:
            if self.mode == "train":
                return None  #Skip

        if len(batch_neg) > 0:
            batch_neg = np.unique(np.concatenate(batch_neg))
        batch_src = np.array(batch_src, dtype="int64")
        batch_dst = np.array(batch_dst, dtype="int64")

        if self.neg_type == "batch_neg":
            batch_neg = batch_dst
        else:
            # TODO user define shape of neg_sample
            neg_shape = batch_dst.shape
            sampled_batch_neg = alias_sample(neg_shape, self.alias, self.events)
            batch_neg = np.concatenate([batch_neg, sampled_batch_neg], 0)

        nodes = np.unique(np.concatenate([batch_src, batch_dst, batch_neg], 0))
        subgraphs = graphsage_sample(self.graph, nodes, self.samples)
        subgraphs[0].node_feat["index"] = subgraphs[0].reindex_to_parrent_nodes(
            subgraphs[0].nodes).astype(np.int64)
        subgraphs[0].node_feat["term_ids"] = self.term_ids[subgraphs[
            0].node_feat["index"]].astype(np.int64)
        feed_dict = {}
        for i in range(self.num_layers):
            numpy_list = self.graphs[i].to_numpy(subgraphs[i])
            for j in range(len(numpy_list)):
                attr = "{}_{}".format(i, self.graphs[i]._graph_attr_holder[j])
                feed_dict[attr] = numpy_list[j]

        # only reindex from first subgraph
        sub_src_idx = subgraphs[0].reindex_from_parrent_nodes(batch_src)
        sub_dst_idx = subgraphs[0].reindex_from_parrent_nodes(batch_dst)
        sub_neg_idx = subgraphs[0].reindex_from_parrent_nodes(batch_neg)

        feed_dict["user_index"] = np.array(sub_src_idx, dtype="int64")
        feed_dict["pos_item_index"] = np.array(sub_dst_idx, dtype="int64")
        feed_dict["neg_item_index"] = np.array(sub_neg_idx, dtype="int64")

        feed_dict["user_real_index"] = np.array(batch_src, dtype="int64")
        feed_dict["pos_item_real_index"] = np.array(batch_dst, dtype="int64")
        return list(feed_dict.values())

    def to_batch(self):
        perm = np.arange(0, len(self.line_examples))
        if self.shuffle:
            np.random.shuffle(perm)
        batch = []
        for idx in perm:
            line_example = self.line_examples[idx]
            batch.append(line_example)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __iter__(self):
        try:
            for batch in self.to_batch():
                if batch is None:
                    continue
                yield self.batch_fn(batch)

        except Exception as e:
            log.exception(e)
