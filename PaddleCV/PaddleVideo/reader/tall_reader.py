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
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle

import random
import numpy as np
import math

import paddle
import paddle.fluid as fluid
import functools

from .reader_utils import DataReader
logger = logging.getLogger(__name__)

random.seed(0)

THREAD = 8
BUF_SIZE = 1024


class TallReader(DataReader):
    """
    Data reader for TALL model, which is processing TACOS dataset and generate a reader iterator for TALL model.
    """
    def __init__(self, name, mode, cfg):
	self.name = name
	self.mode = mode
	self.cfg = cfg	

    def create_reader(self):
        cfg = self.cfg
        mode = self.mode
         
	if self.mode == 'train':
	    train_batch_size = cfg.TRAIN.batch_size
	    return paddle.batch(train(cfg),batch_size = train_batch_size, drop_last=True)
	elif self.mode == 'test':
	    test_batch_size = cfg.TEST.batch_size
	    return paddle.batch(test(cfg),batch_size = test_batch_size, drop_last=True)
	else:
	    logger.info("Not implemented")
	    raise NotImplementedError


    '''
    calculate temporal intersection over union
    '''
    def calculate_IoU(i0, i1):
    	union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    	inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    	iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    	return iou

    '''
    calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
    '''
    #[(x1_max-x1_min)-overlap]/(x1_max-x1_min)
    def calculate_nIoL(base, sliding_clip):
    	inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    	inter_l = inter[1]-inter[0]
    	length = sliding_clip[1]-sliding_clip[0]
    	nIoL = 1.0*(length-inter_l)/length
    	return nIoL

    def get_context_window(sliding_clip_path, clip_name, win_length, context_size, feats_dimen):
    	# compute left (pre) and right (post) context features based on read_unit_level_feats().
    	movie_name = clip_name.split("_")[0]
    	start = int(clip_name.split("_")[1])
    	end = int(clip_name.split("_")[2].split(".")[0])
    	clip_length = context_size
    	left_context_feats = np.zeros([win_length, feats_dimen], dtype=np.float32)
    	right_context_feats = np.zeros([win_length, feats_dimen], dtype=np.float32)
    	last_left_feat = np.load(sliding_clip_path+clip_name)
    	last_right_feat = np.load(sliding_clip_path+clip_name)
    	for k in range(win_length):
            left_context_start = start - clip_length * (k + 1)
            left_context_end = start - clip_length * k
            right_context_start = end + clip_length * k
            right_context_end = end + clip_length * (k + 1)
            left_context_name = movie_name + "_" + str(left_context_start) + "_" + str(left_context_end) + ".npy"
            right_context_name = movie_name + "_" + str(right_context_start) + "_" + str(right_context_end) + ".npy"
            if os.path.exists(sliding_clip_path+left_context_name):
            	left_context_feat = np.load(sliding_clip_path+left_context_name)
            	last_left_feat = left_context_feat
            else:
            	left_context_feat = last_left_feat
            if os.path.exists(sliding_clip_path+right_context_name):
            	right_context_feat = np.load(sliding_clip_path+right_context_name)
            	last_right_feat = right_context_feat
            else:
            	right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)

    def process_data(sample, is_train):
        clip_sentence_pair, sliding_clip_path, context_num, context_size, feats_dimen , sent_vec_dim = sample

        if is_train:
            offset = np.zeros(2, dtype=np.float32)

            clip_name = clip_sentence_pair[0]
            feat_path = sliding_clip_path+clip_sentence_pair[2]
            featmap = np.load(feat_path)
            left_context_feat, right_context_feat = get_context_window(sliding_clip_path, clip_sentence_pair[2], context_num, context_size, feats_dimen)
            image = np.hstack((left_context_feat, featmap, right_context_feat))
            sentence = clip_sentence_pair[1][:sent_vec_dim]
            p_offset = clip_sentence_pair[3]
            l_offset = clip_sentence_pair[4]
            offset[0] = p_offset
            offset[1] = l_offset

            return image, sentence, offset
    	else:
            pass

    def make_train_reader(cfg, clip_sentence_pairs_iou, shuffle=False, is_train=True):
    	sliding_clip_path = cfg.TRAIN.sliding_clip_path
    	context_num = cfg.TRAIN.context_num
    	context_size = cfg.TRAIN.context_size
    	feats_dimen = cfg.TRAIN.feats_dimen
    	sent_vec_dim = cfg.TRAIN.sent_vec_dim

        def reader():
            if shuffle:
            	random.shuffle(clip_sentence_pairs_iou)
            for clip_sentence_pair in clip_sentence_pairs_iou:
            	yield [clip_sentence_pair, sliding_clip_path, context_num, context_size, feats_dimen, sent_vec_dim]
        
        mapper = functools.partial(
            process_data,
            is_train=is_train)

        return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)

    def train(cfg):
    	## TALL
    	feats_dimen = cfg.TRAIN.feats_dimen
    	context_num = cfg.TRAIN.context_num
    	context_size = cfg.TRAIN.context_size
    	visual_feature_dim = cfg.TRAIN.visual_feature_dim
    	sent_vec_dim = cfg.TRAIN.sent_vec_dim
    	sliding_clip_path = cfg.TRAIN.sliding_clip_path
    	cs = pickle.load(open(cfg.TRAIN.train_clip_sentvec))
    	movie_length_info = pickle.load(open(cfg.TRAIN.movie_length_info))
    
    	clip_sentence_pairs = []
        for l in cs:
            clip_name = l[0]
            sent_vecs = l[1]
            for sent_vec in sent_vecs:
                clip_sentence_pairs.append((clip_name, sent_vec)) #10146
        print "TRAIN: " + str(len(clip_sentence_pairs))+" clip-sentence pairs are readed"

    	movie_names_set = set()
   	movie_clip_names = {}
    	# read groundtruth sentence-clip pairs
    	for k in range(len(clip_sentence_pairs)):
            clip_name = clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                movie_clip_names[movie_name] = []
            movie_clip_names[movie_name].append(k)
    	movie_names = list(movie_names_set)
    	num_samples = len(clip_sentence_pairs)
    	print "TRAIN: " + str(len(movie_names))+" movies."

    	# read sliding windows, and match them with the groundtruths to make training samples
    	sliding_clips_tmp = os.listdir(sliding_clip_path) #161396
    	clip_sentence_pairs_iou = []

    	#count = 0
    	for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[2]=="npy":
            	movie_name = clip_name.split("_")[0]
            	for clip_sentence in clip_sentence_pairs:
                    original_clip_name = clip_sentence[0]
                    original_movie_name = original_clip_name.split("_")[0]
                    if original_movie_name==movie_name:
                        start = int(clip_name.split("_")[1])
                        end = int(clip_name.split("_")[2].split(".")[0])
                    	o_start = int(original_clip_name.split("_")[1])
                    	o_end = int(original_clip_name.split("_")[2].split(".")[0])
                    	iou = calculate_IoU((start, end), (o_start, o_end))
                    	if iou>0.5:
                            nIoL=calculate_nIoL((o_start, o_end), (start, end))
                            if nIoL<0.15:
                            	movie_length = movie_length_info[movie_name.split(".")[0]]
                            	start_offset = o_start-start
                            	end_offset = o_end-end
                            	clip_sentence_pairs_iou.append((clip_sentence[0], clip_sentence[1], clip_name, start_offset, end_offset))
    #                           count += 1
    #        if count > 200:
    #   /yield     break
        num_samples_iou = len(clip_sentence_pairs_iou)
    	print "TRAIN: " + str(len(clip_sentence_pairs_iou))+" iou clip-sentence pairs are readed"
    
    	return make_train_reader(cfg, clip_sentence_pairs_iou, shuffle=True, is_train=True)
    
    def test(cfg):
    	test_dataset = TACoS_Test_dataset(cfg)

    	idx = 0
        for movie_name in test_dataset.movie_names:
            idx += 1
            print("%d/%d" % (idx, all_number))

            movie_clip_featmaps, movie_clip_sentences = test_dataset.load_movie_slidingclip(movie_name, 16)
            print("sentences: " + str(len(movie_clip_sentences)))
            print("clips: " + str(len(movie_clip_featmaps)))  # candidate clips)

            sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
            sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])

            for k in range(len(movie_clip_sentences)):
            	sent_vec = movie_clip_sentences[k][1]
            	sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])  # 1,4800
            	#sent_vec = torch.from_numpy(sent_vec).cuda()
            
                for t in range(len(movie_clip_featmaps)):
                    featmap = movie_clip_featmaps[t][1]
                    visual_clip_name = movie_clip_featmaps[t][0]
                
                    start = float(visual_clip_name.split("_")[1])
                    end = float(visual_clip_name.split("_")[2].split("_")[0])
                
                    featmap = np.reshape(featmap, [1, featmap.shape[0]])
                    feed_data = [[featmap, sent_vec, start, end, k, t, movie_clip_sentences, movie_clip_featmaps]]
		    yield feed_data

class TACoS_Test_dataset(cfg):
    '''
    '''
    	def __init__(self, cfg):
            self.context_num = cfg.TEST.context_num 
            self.visual_feature_dim = cfg.TEST.visual_feature_dim
            self.feats_dimen = cfg.TEST.feats_dimen
            self.context_size = cfg.TEST.context_size
            self.semantic_size = cfg.TEST.semantic_size
            self.sliding_clip_path = cfg.TEST.sliding_clip_path
            self.sent_vec_dim = cfg.TEST.sent_vec_dim
            self.cs = cPickle.load(open(cfg.TEST.test_clip_sentvec))
            self.clip_sentence_pairs = []
            for l in self.cs:
            	clip_name = l[0]
            	sent_vecs = l[1]
            	for sent_vec in sent_vecs:
                    self.clip_sentence_pairs.append((clip_name, sent_vec))
            print "TEST: " + str(len(self.clip_sentence_pairs)) + " pairs are readed"
    
            movie_names_set = set()
            self.movie_clip_names = {}
            for k in range(len(self.clip_sentence_pairs)):
                clip_name = self.clip_sentence_pairs[k][0]
                movie_name = clip_name.split("_")[0]
                if not movie_name in movie_names_set:
                    movie_names_set.add(movie_name)
                    self.movie_clip_names[movie_name] = []
                self.movie_clip_names[movie_name].append(k)
            self.movie_names = list(movie_names_set)
            print "TEST: " + str(len(self.movie_names)) + " movies."

            self.clip_num_per_movie_max = 0
            for movie_name in self.movie_clip_names:
                if len(self.movie_clip_names[movie_name])>self.clip_num_per_movie_max: self.clip_num_per_movie_max = len(self.movie_clip_names[movie_name])
            print "TEST: " + "Max number of clips in a movie is "+str(self.clip_num_per_movie_max)

            sliding_clips_tmp = os.listdir(self.sliding_clip_path) # 62741
            self.sliding_clip_names = []
            for clip_name in sliding_clips_tmp:
            	if clip_name.split(".")[2]=="npy":
                    movie_name = clip_name.split("_")[0]
                    if movie_name in self.movie_clip_names:
                    	self.sliding_clip_names.append(clip_name.split(".")[0]+"."+clip_name.split(".")[1])
            self.num_samples = len(self.clip_sentence_pairs)
            print "TEST: " + "sliding clips number: "+str(len(self.sliding_clip_names))

    	def get_test_context_window(self, clip_name, win_length):
        # compute left (pre) and right (post) context features based on read_unit_level_feats().
            movie_name = clip_name.split("_")[0]
            start = int(clip_name.split("_")[1])
            end = int(clip_name.split("_")[2].split(".")[0])
            clip_length = self.context_size #128
            left_context_feats = np.zeros([win_length, self.feats_dimen], dtype=np.float32) #(1,4096)
            right_context_feats = np.zeros([win_length, self.feats_dimen], dtype=np.float32)#(1,4096)
            last_left_feat = np.load(self.sliding_clip_path+clip_name)
            last_right_feat = np.load(self.sliding_clip_path+clip_name)
            for k in range(win_length):
            	left_context_start = start - clip_length * (k + 1)
            	left_context_end = start - clip_length * k
            	right_context_start = end + clip_length * k
            	right_context_end = end + clip_length * (k + 1)
            	left_context_name = movie_name + "_" + str(left_context_start) + "_" + str(left_context_end) + ".npy"
            	right_context_name = movie_name + "_" + str(right_context_start) + "_" + str(right_context_end) + ".npy"
                if os.path.exists(self.sliding_clip_path+left_context_name):
                    left_context_feat = np.load(self.sliding_clip_path+left_context_name)
                    last_left_feat = left_context_feat
            	else:
                    left_context_feat = last_left_feat
            	if os.path.exists(self.sliding_clip_path+right_context_name):
                    right_context_feat = np.load(self.sliding_clip_path+right_context_name)
                    last_right_feat = right_context_feat
            	else:
                    right_context_feat = last_right_feat
                left_context_feats[k] = left_context_feat
                right_context_feats[k] = right_context_feat
            return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)

        def load_movie_slidingclip(self, movie_name, sample_num):
            # load unit level feats and sentence vector
            movie_clip_sentences = []
            movie_clip_featmap = []
            clip_set = set()
            for k in range(len(self.clip_sentence_pairs)):
            	if movie_name in self.clip_sentence_pairs[k][0]:
                    movie_clip_sentences.append((self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))
            for k in range(len(self.sliding_clip_names)):
                if movie_name in self.sliding_clip_names[k]:
                    # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                    visual_feature_path = self.sliding_clip_path+self.sliding_clip_names[k]+".npy"
                    #context_feat=self.get_context(self.sliding_clip_names[k]+".npy")
                    left_context_feat,right_context_feat = self.get_test_context_window(self.sliding_clip_names[k]+".npy",1)
                    feature_data = np.load(visual_feature_path)
                    #comb_feat=np.hstack((context_feat,feature_data))
                    comb_feat = np.hstack((left_context_feat,feature_data,right_context_feat))
                    movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat))
            return movie_clip_featmap, movie_clip_sentences


    
