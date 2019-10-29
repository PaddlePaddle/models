import os
import sys
import time
import numpy as np
import shutil
import json
import argparse
import logging

from config import *
from accuracy_metrics import *
import reader
import paddle
import paddle.fluid as fluid

import pdb

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='TALL',
        help='name of model to train.')
    parser.add_argument(
        '--dataset',
        type=str,
        default='TACoS',
        help='name of dataset to train.')
    args = parser.parse_args()
    return args

args = parse_args()
logger.info(args)

config_path = "tall.yaml"
config = parse_config(config_path)
print_configs(config, 'MODEL')

#MODEL
visual_feature_dim = config.MODEL.visual_feature_dim
semantic_size = config.MODEL.semantic_size
sentence_embedding_size = config.MODEL.sentence_embedding_size
hidden_size = config.MODEL.hidden_size
output_size = config.MODEL.output_size
pretrained_model = None
#pretrained_model = "output/20/"

#TRAIN
epochs = config.TRAIN.epoch
train_batch_size = config.TRAIN.batch_size 
context_size = config.TRAIN.context_size
context_num = config.TRAIN.context_num
feats_dimen = config.TRAIN.feats_dimen
sent_vec_dim = config.TRAIN.sent_vec_dim
off_size = config.TRAIN.off_size
train_sliding_clip_path = config.TRAIN.sliding_clip_path
train_clip_sentvec = config.TRAIN.train_clip_sentvec
movie_length_info = config.TRAIN.movie_length_info

#TEST
test_batch_size = config.TEST.batch_size 
test_sliding_clip_path = config.TEST.sliding_clip_path
test_clip_sentvec = config.TEST.test_clip_sentvec

#OUTPUT
model_save_dir = "output"

bias_attr = fluid.ParamAttr(regularizer=fluid.regularizer.L2Decay(0.0),
        initializer=fluid.initializer.NormalInitializer(scale=0.0))

def TALLModel():
    visual_shape = visual_feature_dim
    sentence_shape = sentence_embedding_size
    offset_shape = off_size

    images = fluid.layers.data(
        name='train_visual',
        shape=[visual_shape],
        dtype='float32',
        lod_level=0)
    sentences = fluid.layers.data(
        name='train_sentences',
        shape=[sentence_shape],
        dtype='float32',
        lod_level=0)
    offsets = fluid.layers.data(
        name='train_offsets',
        shape=[offset_shape],
        dtype='float32')

    # visual2semantic
    transformed_clip_train = fluid.layers.fc(
        input=images,
        size=semantic_size,
        act=None,
        name='v2s_lt',
        param_attr=fluid.ParamAttr(
            name='v2s_lt_weights',
            initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)),
        bias_attr=False)
    #l2_normalize
    transformed_clip_train = fluid.layers.l2_normalize(x=transformed_clip_train, axis=1)
    # sentenct2semantic
    transformed_sentence_train = fluid.layers.fc(
        input=sentences,
        size=semantic_size,
        act=None,
        name='s2s_lt',
        param_attr=fluid.ParamAttr(
            name='s2s_lt_weights',
            initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)),
        bias_attr=False)
    #l2_normalize
    transformed_sentence_train = fluid.layers.l2_normalize(x=transformed_sentence_train, axis=1)
    
    def cross_modal_comb(visual_feat, sentence_embed):
        #batch_size = visual_feat.size(0)
        visual_feat = fluid.layers.reshape(visual_feat, [1, -1, semantic_size])
        vv_feature = fluid.layers.expand(visual_feat, [train_batch_size, 1, 1])
        #vv_feature[0,:,:] == vv_feature[1,:,:]
        sentence_embed = fluid.layers.reshape(sentence_embed, [-1, 1, semantic_size])
        ss_feature = fluid.layers.expand(sentence_embed, [1, train_batch_size, 1])
        #ss_feature[:,0,:] == ss_feature[:,1,:]

        concat_feature = fluid.layers.concat([vv_feature, ss_feature], axis = 2) #1,1,2048

        #vv_feature = fluid.layers.Print(vv_feature, message='vv_feature',
        #                       summarize=10)
        #ss_feature = fluid.layers.Print(ss_feature, message='ss_feature',
        #                        summarize=10)
        mul_feature = vv_feature * ss_feature # B,B,1024
        add_feature = vv_feature + ss_feature # B,B,1024

        comb_feature = fluid.layers.concat([mul_feature, add_feature, concat_feature], axis = 2)
        return comb_feature

    cross_modal_vec_train=cross_modal_comb(transformed_clip_train, transformed_sentence_train)
    cross_modal_vec_train=fluid.layers.unsqueeze(input=cross_modal_vec_train, axes=[0])
    cross_modal_vec_train=fluid.layers.transpose(cross_modal_vec_train, perm=[0, 3, 1, 2])
    
    mid_output = fluid.layers.conv2d(
        input=cross_modal_vec_train,
        num_filters=hidden_size,
        filter_size=1,
        stride=1,
        act="relu",
        param_attr=fluid.param_attr.ParamAttr(name="mid_out_weights"),
        bias_attr=False)

    sim_score_mat_train = fluid.layers.conv2d(
        input=mid_output,
        num_filters=output_size,
        filter_size=1,
        stride=1,
        act=None,
        param_attr=fluid.param_attr.ParamAttr(name="sim_mat_weights"),
        bias_attr=False)
    sim_score_mat_train = fluid.layers.squeeze(input=sim_score_mat_train, axes=[0])

    return sim_score_mat_train, offsets

def train_model():
    outs, offs = TALLModel()
    sim_score_mat = outs[0]
    p_reg_mat = outs[1]
    l_reg_mat = outs[2]
    # loss cls, not considering iou
    input_size = outs.shape[1]
    I = fluid.layers.diag(np.array([1]*input_size).astype('float32'))
    I_2 = -2 * I
    all1 = fluid.layers.ones(shape=[input_size,input_size], dtype="float32")

    mask_mat = I_2 + all1
    #               | -1  1   1...   |
    #   mask_mat =  | 1  -1   1...   |
    #               | 1   1  -1 ...  |

    alpha = 1.0 / input_size
    lambda_regression = 0.01
    batch_para_mat = alpha * all1
    para_mat = I + batch_para_mat

    sim_mask_mat = fluid.layers.exp(mask_mat*sim_score_mat)
    loss_mat = fluid.layers.log(all1 + sim_mask_mat)
    loss_mat = loss_mat*para_mat
    loss_align = fluid.layers.mean(loss_mat)
    
    # regression loss
    reg_ones = fluid.layers.ones(shape=[input_size, 1], dtype="float32")
    l_reg_diag = fluid.layers.matmul(l_reg_mat*I, reg_ones, transpose_x=True, transpose_y=False)
    p_reg_diag = fluid.layers.matmul(p_reg_mat*I, reg_ones, transpose_x=True, transpose_y=False)
    #l_reg_diag = (l_reg_mat*I) * reg_ones
    #p_reg_diag = (p_reg_mat*I) * reg_ones
    offset_pred = fluid.layers.concat(input=[p_reg_diag, l_reg_diag], axis=1)
    loss_reg = fluid.layers.mean(fluid.layers.abs(offset_pred - offs)) # L1 loss
    loss = lambda_regression*loss_reg +loss_align
    avg_loss = fluid.layers.mean(loss)

    return avg_loss

def optimizer_func():
    fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))
    #lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(hidden_size, 1000)
    
    return fluid.optimizer.Adam(
            learning_rate=1e-3)
            #regularization=fluid.regularizer.L2DecayRegularizer(
            #    regularization_coeff=1e-3))

def train():
    train_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            avg_cost = train_model()
            optimizer = optimizer_func()
            optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    
    train_exe = fluid.ParallelExecutor(main_program=train_prog,
            use_cuda=True, loss_name=avg_cost.name)
    train_reader = paddle.batch(reader.train(config), batch_size=train_batch_size, drop_last=True)
    feeder = fluid.DataFeeder(place=place, program=train_prog,
            feed_list=['train_visual', 'train_sentences', 'train_offsets'])
    
    train_fetch_list = [avg_cost.name]

    def save_model(postfix):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print ('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=train_prog)

    for pass_id in range(epochs):
        for iter_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss = train_exe.run(fetch_list=train_fetch_list, feed=feeder.feed(data), return_numpy=True)
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss[0]))
            str_time = time.strftime('%m-%d_%H:%M:%S', time.localtime())
            if iter_id % 10 == 0:
                print ('[' + str_time +
                        '] [TRAIN] Pass: {0}\ttrainbatch: {1}\tloss: {2}\ttime: {3}'
                        .format(pass_id, iter_id, '%.6f'%loss, '%2.2f sec'%period))

        save_model(str(pass_id))
        if pass_id % 20 == 0 and pass_id > 0:
            test("{}/{}".format(model_save_dir, pass_id))

def test_model():
    visual_shape = visual_feature_dim
    sentence_shape = sentence_embedding_size

    images = fluid.layers.data(
        name='test_visual',
        shape=[visual_shape],
        dtype='float32',
        lod_level=0)
    sentences = fluid.layers.data(
        name='test_sentences',
        shape=[sentence_shape],
        dtype='float32',
        lod_level=0)

    # visual2semantic
    transformed_clip_test = fluid.layers.fc(
        input=images,
        size=semantic_size,
        act=None,
        name='v2s_lt',
        param_attr=fluid.ParamAttr(
            name='v2s_lt_weights',
            initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)),
        bias_attr=False)
    #l2_normalize
    transformed_clip_test = fluid.layers.l2_normalize(x=transformed_clip_test, axis=1)
    # sentenct2semantic
    transformed_sentence_test = fluid.layers.fc(
        input=sentences,
        size=semantic_size,
        act=None,
        name='s2s_lt',
        param_attr=fluid.ParamAttr(
            name='s2s_lt_weights',
            initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)),
        bias_attr=False)
    #l2_normalize
    transformed_sentence_test = fluid.layers.l2_normalize(x=transformed_sentence_test, axis=1)
    
    def cross_modal_comb(visual_feat, sentence_embed):
        #batch_size = visual_feat.size(0)
        visual_feat = fluid.layers.reshape(visual_feat, [1, -1, semantic_size])
        vv_feature = fluid.layers.expand(visual_feat, [test_batch_size, 1, 1])
        #vv_feature[0,:,:] == vv_feature[1,:,:]
        sentence_embed = fluid.layers.reshape(sentence_embed, [-1, 1, semantic_size])
        ss_feature = fluid.layers.expand(sentence_embed, [1, test_batch_size, 1])
        #ss_feature[:,0,:] == ss_feature[:,1,:]

        concat_feature = fluid.layers.concat([vv_feature, ss_feature], axis = 2) #1,1,2048

        #vv_feature = fluid.layers.Print(vv_feature, message='vv_feature',
        #                       summarize=10)
        #ss_feature = fluid.layers.Print(ss_feature, message='ss_feature',
        #                        summarize=10)
        mul_feature = vv_feature * ss_feature # B,B,1024
        add_feature = vv_feature + ss_feature # B,B,1024

        comb_feature = fluid.layers.concat([mul_feature, add_feature, concat_feature], axis = 2)
        return comb_feature

    cross_modal_vec_test=cross_modal_comb(transformed_clip_test, transformed_sentence_test)
    cross_modal_vec_test=fluid.layers.unsqueeze(input=cross_modal_vec_test, axes=[0])
    cross_modal_vec_test=fluid.layers.transpose(cross_modal_vec_test, perm=[0, 3, 1, 2])
    
    mid_output = fluid.layers.conv2d(
        input=cross_modal_vec_test,
        num_filters=hidden_size,
        filter_size=1,
        stride=1,
        act="relu",
        param_attr=fluid.param_attr.ParamAttr(name="mid_out_weights"),
        bias_attr=False)

    sim_score_mat_test = fluid.layers.conv2d(
        input=mid_output,
        num_filters=output_size,
        filter_size=1,
        stride=1,
        act=None,
        param_attr=fluid.param_attr.ParamAttr(name="sim_mat_weights"),
        bias_attr=False)
    sim_score_mat_test = fluid.layers.squeeze(input=sim_score_mat_test, axes=[0])

    return sim_score_mat_test

def test(model_best):
    global best_R1_IOU5
    global best_R5_IOU5
    global best_R1_IOU5_epoch
    global best_R5_IOU5_epoch

    IoU_thresh = [0.1, 0.3, 0.5, 0.7]
    all_correct_num_10 = [0.0] * 5
    all_correct_num_5 = [0.0] * 5
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0
    
    test_dataset = reader.TACoS_Test_dataset(config)
    all_number = len(test_dataset.movie_names)

    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            outputs = test_model()
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    
    if model_best:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        #fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)
        fluid.io.load_params(exe, model_best, main_program=test_prog)

    feeder = fluid.DataFeeder(place=place, program=test_prog,
            feed_list=['test_visual', 'test_sentences'])

    test_fetch_list = [outputs.name]

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
                feed_data = [[featmap, sent_vec]]

                # forward
                outputs = exe.run(test_prog, feed=feeder.feed(feed_data),
                    fetch_list=test_fetch_list, return_numpy=True)
                outputs = np.squeeze(outputs)

                # TALL network
                sentence_image_mat[k, t] = outputs[0]

                # sentence_image_mat[k, t] = expit(outputs[0]) * conf_score
                reg_end = end + outputs[2]
                reg_start = start + outputs[1]

                sentence_image_reg_mat[k, t, 0] = reg_start
                sentence_image_reg_mat[k, t, 1] = reg_end

        iclips = [b[0] for b in movie_clip_featmaps]
        sclips = [b[0] for b in movie_clip_sentences]

        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            print(movie_name + " IoU=" + str(IoU) + ", R@10: " + str(correct_num_10 / len(sclips)) + "; IoU=" + str(IoU) + ", R@5: " + str(correct_num_5 / len(sclips)) + "; IoU=" + str(IoU) + ", R@1: " + str(correct_num_1 / len(sclips)))

            all_correct_num_10[k] += correct_num_10
            all_correct_num_5[k] += correct_num_5
            all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)
        
    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@5: " + str(all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))
        
    R1_IOU5 = all_correct_num_1[2] / all_retrievd
    R5_IOU5 = all_correct_num_5[2] / all_retrievd

    print "{}\n".format("best_R1_IOU5: %0.3f" % R1_IOU5)
    print "{}\n".format("best_R5_IOU5: %0.3f" % R5_IOU5)

def main():
    train()
    test("checkpoints/20/")

if __name__ == '__main__':
    main()
