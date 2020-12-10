import numpy as np
import pandas as pd
import os
import random
import paddle.fluid as fluid
from youtubednn import YoutubeDNN
import paddle
import args
import logging
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def train(args):
    youtube_model = YoutubeDNN()
    inputs = youtube_model.input_data(args.watch_vec_size, args.search_vec_size, args.other_feat_size)
    loss, acc, l3 = youtube_model.net(inputs, args.output_size, layers=[128, 64, 32])

    sgd = fluid.optimizer.SGD(learning_rate=args.base_lr)
    sgd.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # Build a random data set.
    sample_size = 100
    watch_vecs = []
    search_vecs = []
    other_feats = []
    labels = []

    for i in range(sample_size):
        watch_vec = np.random.rand(args.batch_size, args.watch_vec_size)
        search_vec = np.random.rand(args.batch_size, args.search_vec_size)
        other_feat = np.random.rand(args.batch_size, args.other_feat_size)
        watch_vecs.append(watch_vec)
        search_vecs.append(search_vec)
        other_feats.append(other_feat)
        label = np.random.randint(args.output_size, size=(args.batch_size, 1))
        labels.append(label)
    for epoch in range(args.epochs):
        for i in range(sample_size):
            begin = time.time()
            loss_data, acc_val = exe.run(fluid.default_main_program(),
                                feed={
                                    "watch_vec": watch_vecs[i].astype('float32'),
                                    "search_vec": search_vecs[i].astype('float32'),
                                    "other_feat": other_feats[i].astype('float32'),
                                    "label": np.array(labels[i]).reshape(args.batch_size, 1)
                                },
                                return_numpy=True,
                                fetch_list=[loss.name, acc.name])
            end = time.time()
            logger.info("epoch_id: {}, batch_time: {:.5f}s, loss: {:.5f}, acc: {:.5f}".format(
                epoch, end-begin, float(np.array(loss_data)), np.array(acc_val)[0]))
        #save model
        model_dir = os.path.join(args.model_dir, 'epoch_' + str(epoch + 1), "checkpoint")

        feed_var_names = ["watch_vec", "search_vec", "other_feat"]
        fetch_vars = [l3]
        fluid.io.save_inference_model(model_dir, feed_var_names, fetch_vars, exe)

    #save all video vector
    video_array = np.array(fluid.global_scope().find_var('l4_weight').get_tensor())
    video_vec = pd.DataFrame(video_array)
    video_vec.to_csv(args.video_vec_path, mode="a", index=False, header=0)

if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = args.parse_args()
    if(os.path.exists(args.video_vec_path)):
        os.system("rm " + args.video_vec_path)
    train(args)
