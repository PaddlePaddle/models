import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import paddle.fluid as fluid
import os
from gmf import GMF
from mlp import MLP
from neumf import NeuMF
from Dataset import Dataset
import logging
import paddle
import args
import utils
import time

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
_args = None
_model_path = None

def run_infer(args, model_path, test_data_path):
    test_data_generator = utils.CriteoDataset()
    
    with fluid.scope_guard(fluid.Scope()):
        test_reader = paddle.batch(test_data_generator.test(test_data_path, False), batch_size=args.test_batch_size)
            
        place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(model_path, exe)

        for data in test_reader():
            user_input = np.array([dat[0] for dat in data])
            item_input = np.array([dat[1] for dat in data])

            pred_val = exe.run(infer_program,
                       feed={"user_input": user_input,
                            "item_input": item_input},
                       fetch_list=fetch_vars,
                        return_numpy=True)
        
            return pred_val[0].reshape(1, -1).tolist()[0]

def evaluate_model(args, testRatings, testNegatives, K, model_path):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _model_path 
    global _args
    
    _args = args
    _model_path= model_path
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs = [],[]
    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    users = users.reshape(-1,1)
    items_array = np.array(items).reshape(-1,1)
    temp = np.hstack((users, items_array))
    np.savetxt("Data/test.txt", temp, fmt='%d', delimiter=',')
    predictions = run_infer(_args, _model_path, _args.test_data_path)

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)

    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
