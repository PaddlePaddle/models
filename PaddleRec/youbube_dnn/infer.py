import paddle.fluid as fluid
import numpy as np
import pandas as pd
import time
import sys
import os
import args
import logging
from youtubednn import YoutubeDNN

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def infer(args):
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    cur_model_path = os.path.join(args.model_dir, 'epoch_' + str(args.test_epoch), "checkpoint")

    with fluid.scope_guard(fluid.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(cur_model_path, exe)
        # Build a random data set.
        sample_size = 100
        watch_vecs = []
        search_vecs = []
        other_feats = []

        for i in range(sample_size):
            watch_vec = np.random.rand(1, args.watch_vec_size)
            search_vec = np.random.rand(1, args.search_vec_size)
            other_feat = np.random.rand(1, args.other_feat_size)
            watch_vecs.append(watch_vec)
            search_vecs.append(search_vec)
            other_feats.append(other_feat)

        for i in range(sample_size):
            l3 = exe.run(infer_program,
                        feed={
                            "watch_vec": watch_vecs[i].astype('float32'),
                            "search_vec": search_vecs[i].astype('float32'),
                            "other_feat": other_feats[i].astype('float32'),
                        },
                        return_numpy=True,
                        fetch_list=fetch_vars)

            user_vec = pd.DataFrame(l3[0])
            user_vec.to_csv(args.user_vec_path, mode="a", index=False, header=0)

if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = args.parse_args()
    if(os.path.exists(args.user_vec_path)):
        os.system("rm " + args.user_vec_path)
    infer(args)