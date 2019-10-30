#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import random
import sys
import numpy as np
import h5py
import multiprocessing
import functools
import paddle

random.seed(0)

import logging
logger = logging.getLogger(__name__)

try:
    import cPickle as pickle
except:
    import pickle

from .reader_utils import DataReader

python_ver = sys.version_info


class TALLReader(DataReader):
    """
    Data reader for TALL model, which was stored as features extracted by prior networks
    """

    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode

        self.visual_feature_dim = cfg.MODEL.visual_feature_dim
        self.movie_length_info = cfg.TRAIN.movie_length_info

        self.feats_dimen = cfg[mode.upper()]['feats_dimen']
        self.context_num = cfg[mode.upper()]['context_num']
        self.context_size = cfg[mode.upper()]['context_size']
        self.sent_vec_dim = cfg[mode.upper()]['sent_vec_dim']
        self.sliding_clip_path = cfg[mode.upper()]['sliding_clip_path']
        self.clip_sentvec = cfg[mode.upper()]['clip_sentvec']
        self.semantic_size = cfg[mode.upper()]['semantic_size']

        self.batch_size = cfg[mode.upper()]['batch_size']
        self.init_data()

    def get_context_window(self, clip_name):
        # compute left (pre) and right (post) context features based on read_unit_level_feats().
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        clip_length = self.context_size
        left_context_feats = np.zeros(
            [self.context_num, self.feats_dimen], dtype=np.float32)
        right_context_feats = np.zeros(
            [self.context_num, self.feats_dimen], dtype=np.float32)
        last_left_feat = np.load(
            os.path.join(self.sliding_clip_path, clip_name))
        last_right_feat = np.load(
            os.path.join(self.sliding_clip_path, clip_name))
        for k in range(self.context_num):
            left_context_start = start - clip_length * (k + 1)
            left_context_end = start - clip_length * k
            right_context_start = end + clip_length * k
            right_context_end = end + clip_length * (k + 1)
            left_context_name = movie_name + "_" + str(
                left_context_start) + "_" + str(left_context_end) + ".npy"
            right_context_name = movie_name + "_" + str(
                right_context_start) + "_" + str(right_context_end) + ".npy"
            if os.path.exists(
                    os.path.join(self.sliding_clip_path, left_context_name)):
                left_context_feat = np.load(
                    os.path.join(self.sliding_clip_path, left_context_name))
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(
                    os.path.join(self.sliding_clip_path, right_context_name)):
                right_context_feat = np.load(
                    os.path.join(self.sliding_clip_path, right_context_name))
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(
            left_context_feats, axis=0), np.mean(
                right_context_feats, axis=0)

    def init_data(self):
        def calculate_IoU(i0, i1):
            # calculate temporal intersection over union
            union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
            inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
            iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
            return iou

        def calculate_nIoL(base, sliding_clip):
            # calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
            inter = (max(base[0], sliding_clip[0]), min(base[1],
                                                        sliding_clip[1]))
            inter_l = inter[1] - inter[0]
            length = sliding_clip[1] - sliding_clip[0]
            nIoL = 1.0 * (length - inter_l) / length
            return nIoL

        # load file
        if (self.mode == 'train') or (self.mode == 'valid'):
            if python_ver < (3, 0):
                cs = pickle.load(open(self.clip_sentvec, 'rb'))
                movie_length_info = pickle.load(
                    open(self.movie_length_info, 'rb'))
            else:
                cs = pickle.load(
                    open(self.clip_sentvec, 'rb'), encoding='bytes')
                movie_length_info = pickle.load(
                    open(self.movie_length_info, 'rb'), encoding='bytes')
        elif (self.mode == 'test') or (self.mode == 'infer'):
            if python_ver < (3, 0):
                cs = pickle.load(open(self.clip_sentvec, 'rb'))
            else:
                cs = pickle.load(
                    open(self.clip_sentvec, 'rb'), encoding='bytes')

        self.clip_sentence_pairs = []
        for l in cs:
            clip_name = l[0].decode('utf-8')  #byte object to string
            sent_vecs = l[1]  #numpy array
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))  #10146
        logger.info(self.mode.upper() + ':' + str(
            len(self.clip_sentence_pairs)) + " clip-sentence pairs are read")

        movie_names_set = set()
        movie_clip_names = {}
        # read groundtruth sentence-clip pairs
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                movie_clip_names[movie_name] = []
            movie_clip_names[movie_name].append(k)
        self.movie_names = list(movie_names_set)
        logger.info(self.mode.upper() + ':' + str(len(self.movie_names)) +
                    " movies.")

        # read sliding windows, and match them with the groundtruths to make training samples
        sliding_clips_tmp = os.listdir(self.sliding_clip_path)  #161396
        self.clip_sentence_pairs_iou = []
        if self.mode == 'valid':
            # TALL model doesn't take validation during training, it will test after all the training epochs finish.
            return
        if self.mode == 'train':
            num_sliding_clips = len(sliding_clips_tmp)
            count = 0
            for clip_name in sliding_clips_tmp:
                count += 1
                logger.info('processing data ............' + str(count) + '/' +
                            str(num_sliding_clips))
                if clip_name.split(".")[2] == "npy":
                    movie_name = clip_name.split("_")[0]
                    for clip_sentence in self.clip_sentence_pairs:
                        original_clip_name = clip_sentence[0]
                        original_movie_name = original_clip_name.split("_")[0]
                        if original_movie_name == movie_name:
                            start = int(clip_name.split("_")[1])
                            end = int(clip_name.split("_")[2].split(".")[0])
                            o_start = int(original_clip_name.split("_")[1])
                            o_end = int(
                                original_clip_name.split("_")[2].split(".")[0])
                            iou = calculate_IoU((start, end), (o_start, o_end))
                            if iou > 0.5:
                                nIoL = calculate_nIoL((o_start, o_end),
                                                      (start, end))
                                if nIoL < 0.15:
                                    movie_length = movie_length_info[
                                        movie_name.split(".")[0].encode(
                                            'utf-8')]  #str to byte
                                    start_offset = o_start - start
                                    end_offset = o_end - end
                                    self.clip_sentence_pairs_iou.append(
                                        (clip_sentence[0], clip_sentence[1],
                                         clip_name, start_offset, end_offset))
            logger.info('TRAIN:' + str(len(self.clip_sentence_pairs_iou)) +
                        " iou clip-sentence pairs are read")

        elif (self.mode == 'test') or (self.mode == 'infer'):
            for clip_name in sliding_clips_tmp:
                if clip_name.split(".")[2] == "npy":
                    movie_name = clip_name.split("_")[0]
                    if movie_name in movie_clip_names:
                        self.clip_sentence_pairs_iou.append(
                            clip_name.split(".")[0] + "." + clip_name.split(".")
                            [1])

            logger.info('TEST:' + str(len(self.clip_sentence_pairs_iou)) +
                        " iou clip-sentence pairs are read")

    def load_movie_slidingclip(self, clip_sentence_pairs,
                               clip_sentence_pairs_iou, movie_name):
        # load unit level feats and sentence vector
        movie_clip_sentences = []
        movie_clip_featmap = []
        for k in range(len(clip_sentence_pairs)):
            if movie_name in clip_sentence_pairs[k][0]:
                movie_clip_sentences.append(
                    (clip_sentence_pairs[k][0],
                     clip_sentence_pairs[k][1][:self.semantic_size]))
        for k in range(len(clip_sentence_pairs_iou)):
            if movie_name in clip_sentence_pairs_iou[k]:
                visual_feature_path = os.path.join(
                    self.sliding_clip_path, clip_sentence_pairs_iou[k] + ".npy")
                left_context_feat, right_context_feat = self.get_context_window(
                    clip_sentence_pairs_iou[k] + ".npy")
                feature_data = np.load(visual_feature_path)
                comb_feat = np.hstack(
                    (left_context_feat, feature_data, right_context_feat))
                movie_clip_featmap.append(
                    (clip_sentence_pairs_iou[k], comb_feat))
        return movie_clip_featmap, movie_clip_sentences

    def create_reader(self):
        """reader creator for ets model"""
        if self.mode == 'infer':
            return self.make_infer_reader()
        else:
            return self.make_reader()

    def make_infer_reader(self):
        """reader for inference"""

        def reader():
            batch_out = []
            idx = 0
            for movie_name in self.movie_names:
                idx += 1
                movie_clip_featmaps, movie_clip_sentences = self.load_movie_slidingclip(
                    self.clip_sentence_pairs, self.clip_sentence_pairs_iou,
                    movie_name)
                for k in range(len(movie_clip_sentences)):
                    sent_vec = movie_clip_sentences[k][1]
                    sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])
                    for t in range(len(movie_clip_featmaps)):
                        featmap = movie_clip_featmaps[t][1]
                        visual_clip_name = movie_clip_featmaps[t][0]
                        start = float(visual_clip_name.split("_")[1])
                        end = float(
                            visual_clip_name.split("_")[2].split("_")[0])
                        featmap = np.reshape(featmap, [1, featmap.shape[0]])

                        batch_out.append((featmap, sent_vec, start, end, k, t,
                                          movie_clip_sentences,
                                          movie_clip_featmaps, movie_name))
                        if len(batch_out) == self.batch_size:
                            yield batch_out
                            batch_out = []

        return reader

    def make_reader(self):
        def reader():
            batch_out = []
            if self.mode == 'valid':
                return
            elif self.mode == 'train':
                random.shuffle(self.clip_sentence_pairs_iou)
                for clip_sentence_pair in self.clip_sentence_pairs_iou:
                    offset = np.zeros(2, dtype=np.float32)
                    clip_name = clip_sentence_pair[0]
                    feat_path = os.path.join(self.sliding_clip_path,
                                             clip_sentence_pair[2])
                    featmap = np.load(feat_path)
                    left_context_feat, right_context_feat = self.get_context_window(
                        clip_sentence_pair[2])
                    image = np.hstack(
                        (left_context_feat, featmap, right_context_feat))
                    sentence = clip_sentence_pair[1][:self.sent_vec_dim]
                    p_offset = clip_sentence_pair[3]
                    l_offset = clip_sentence_pair[4]
                    offset[0] = p_offset
                    offset[1] = l_offset
                    batch_out.append((image, sentence, offset))
                    if len(batch_out) == self.batch_size:
                        yield batch_out
                        batch_out = []

            elif self.mode == 'test':
                for movie_name in self.movie_names:
                    movie_clip_featmaps, movie_clip_sentences = self.load_movie_slidingclip(
                        self.clip_sentence_pairs, self.clip_sentence_pairs_iou,
                        movie_name)
                    for k in range(len(movie_clip_sentences)):
                        sent_vec = movie_clip_sentences[k][1]
                        sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])
                        for t in range(len(movie_clip_featmaps)):
                            featmap = movie_clip_featmaps[t][1]
                            visual_clip_name = movie_clip_featmaps[t][0]
                            start = float(visual_clip_name.split("_")[1])
                            end = float(
                                visual_clip_name.split("_")[2].split("_")[0])
                            featmap = np.reshape(featmap, [1, featmap.shape[0]])

                            batch_out.append((featmap, sent_vec, start, end, k,
                                              t, movie_clip_sentences,
                                              movie_clip_featmaps, movie_name))
                            if len(batch_out) == self.batch_size:
                                yield batch_out
                                batch_out = []
            else:
                raise NotImplementedError('mode {} not implemented'.format(
                    self.mode))

        return reader
