import numpy as np
import os
import paddle.fluid as fluid
from gmf import GMF
from mlp import MLP
from neumf import NeuMF
from Dataset import Dataset
from evaluate import evaluate_model
import logging
import paddle
import args
import utils
import time
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    import paddle
    paddle.enable_static()
  
    args = args.parse_args()
    dataset = Dataset(args.path + args.dataset)
    testRatings, testNegatives = dataset.testRatings, dataset.testNegatives
    topK = 10
   
    begin = time.time()
    model_path = args.model_dir + "/epoch_" + str(12)
    (hits, ndcgs) = evaluate_model(args, testRatings, testNegatives, topK, model_path)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    end = time.time()
    
    logger.info("epoch: {}, epoch_time: {:.5f}s, HR: {:.5f}, NDCG: {:.5f}".format(args.epochs, end - begin, hr, ndcg))
        