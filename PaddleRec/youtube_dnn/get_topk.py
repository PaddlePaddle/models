import numpy as np
import pandas as pd
import args
import copy

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def get_topK(args):
    video_vec = pd.read_csv(args.video_vec_path, header=None)
    user_vec = pd.read_csv(args.user_vec_path, header=None)

    user_video_sim_list = []
    for i in range(user_vec.shape[0]):
        for j in range(video_vec.shape[1]):    
            user_video_sim = cos_sim(np.array(user_vec.loc[i]), np.array(video_vec[j]))
            user_video_sim_list.append(user_video_sim)

        tmp_list=copy.deepcopy(user_video_sim_list)
        tmp_list.sort()
        max_sim_index=[user_video_sim_list.index(one) for one in tmp_list[::-1][:args.topk]]

        print("user:{0}, top K videos:{1}".format(i, max_sim_index))
        user_video_sim_list = []

if __name__ == "__main__":
    args = args.parse_args()
    get_topK(args)