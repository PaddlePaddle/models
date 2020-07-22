import paddle.fluid as fluid
import numpy as np
import sys
import args
import logging
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def fc(tag, data, out_dim, active='prelu'):

        xavier=fluid.initializer.Xavier(uniform=True, fan_in=data.shape[1], fan_out=out_dim)
      
        out = fluid.layers.fc(input=data,
                            size=out_dim,
                            act=active,
                            param_attr=xavier, 
                            bias_attr =xavier,
                            name=tag)
        return out
        
def model(TRIGRAM_D = 1000, L1_N = 300, L2_N = 300, L3_N = 128, Neg = 4):
    query = fluid.data(name="query", shape=[-1, TRIGRAM_D], dtype="float32")
    doc_pos = fluid.data(name="doc_pos", shape=[-1, TRIGRAM_D], dtype="float32")
    doc_negs = [fluid.data(name="doc_neg_" + str(i), shape=[-1, TRIGRAM_D], dtype="float32") for i in range(Neg)]
    
    active = 'tanh'
    query_l1 = fc('query_l1', query, L1_N, active)
    doc_pos_l1 = fc('doc_pos_l1', doc_pos, L1_N, active)

    query_l2 = fc('query_l2', query_l1, L2_N, active)
    doc_pos_l2 = fc('doc_l2', doc_pos_l1, L2_N, active)
    
    query_l3 = fc('query_l3', query_l2, L3_N, active)
    doc_pos_l3 = fc('doc_l3', doc_pos_l2, L3_N, active)
    
    neg_doc_sems = []
    for i, doc_neg in enumerate(doc_negs):
        doc_neg_l1 = fc('doc_neg_l1_' + str(i), doc_neg, L1_N, active)
        doc_neg_l2 = fc('doc_neg_l2_' + str(i), doc_neg_l1, L2_N, active)
        doc_neg_l3 = fc('doc_neg_l3_' + str(i), doc_neg_l2, L3_N, active)
        
        neg_doc_sems.append(doc_neg_l3)
        
    R_Q_D_p  = fluid.layers.cos_sim(query_l3, doc_pos_l3)
    R_Q_D_ns = [fluid.layers.cos_sim(query_l3, neg_doc_sem) for neg_doc_sem in neg_doc_sems]
    
    concat_Rs = fluid.layers.concat(input=[R_Q_D_p] + R_Q_D_ns, axis=-1)
    prob = fluid.layers.softmax(concat_Rs, axis=1)
    hit_prob = fluid.layers.slice(prob, axes=[0,1], starts=[0,0], ends=[args.batch_size, 1])

    loss  = -fluid.layers.reduce_sum(fluid.layers.log(hit_prob))
    avg_loss = fluid.layers.mean(x=loss)

    return avg_loss, R_Q_D_p, [query] + [doc_pos] + doc_negs
    
args = args.parse_args()
loss,R_Q_D_p, data_list = model(args.TRIGRAM_D,args.L1_N,args.L2_N,args.L3_N,args.Neg)

sgd = fluid.optimizer.SGD(learning_rate=args.base_lr)
sgd.minimize(loss)

place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# Build a random data set.
sample_size = 100
l_Qs = []
pos_l_Ds = []


for i in range(sample_size):
    l_Q = np.random.rand(args.batch_size, args.TRIGRAM_D)
    l_Qs.append(l_Q)
    
    l_D = np.random.rand(args.batch_size, args.TRIGRAM_D)
    pos_l_Ds.append(l_D)

neg_l_Ds = [[] for i in range(args.Neg)]
for i in range(sample_size):
    possibilities = list(range(sample_size))
    possibilities.remove(i)
    negatives = np.random.choice(possibilities, args.Neg, replace = False)
    for j in range(args.Neg):
        negative = negatives[j]
        neg_l_Ds[j].append(pos_l_Ds[negative])

for i in range(sample_size):
    begin = time.time()
    loss_data = exe.run(fluid.default_main_program(),
                         feed={
                             "query": l_Qs[i].astype('float32').reshape(args.batch_size,args.TRIGRAM_D),
                             "doc_pos": pos_l_Ds[i].astype('float32').reshape(args.batch_size,args.TRIGRAM_D),
                             "doc_neg_0": neg_l_Ds[0][i].astype('float32'),
                             "doc_neg_1": neg_l_Ds[1][i].astype('float32'),
                             "doc_neg_2": neg_l_Ds[2][i].astype('float32'),
                             "doc_neg_3": neg_l_Ds[3][i].astype('float32'),
                         },
                         return_numpy=True,
                         fetch_list=[loss.name])

    end = time.time()
    logger.info("epoch_id: {}, batch_time: {:.5f}s, loss: {:.5f}".format(i, end-begin, float(np.array(loss_data))))
    
feed_var_names = ["query", "doc_pos"]
fetch_vars = [R_Q_D_p]
fluid.io.save_inference_model(args.model_dir, feed_var_names, fetch_vars, exe)









