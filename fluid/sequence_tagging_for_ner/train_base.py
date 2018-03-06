import paddle.v2 as paddle
import paddle.v2.fluid as fluid
from network_conf_base import ner_net
from utils import logger, load_dict, get_embedding
import reader
import os
import math
import numpy as np

def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

def test(exe, chunk_evaluator, inference_program, test_data, place):
    chunk_evaluator.reset(exe)
    for data in test_data():
        word = to_lodtensor(map(lambda x:x[0], data), place)
        mark = to_lodtensor(map(lambda x:x[1], data), place)
        target = to_lodtensor(map(lambda x:x[2], data), place)
        acc = exe.run(inference_program,
                      feed={"word": word,
                            "mark": mark,
                            "target": target})
    return  chunk_evaluator.eval(exe)

def main(train_data_file,
         test_data_file,
         vocab_file,
         target_file,
         emb_file,
         model_save_dir,
         num_passes=100,
         batch_size=64):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    word_dict = load_dict(vocab_file)
    label_dict = load_dict(target_file)

    word_vector_values = get_embedding(emb_file)

    word_dict_len = len(word_dict)
    label_dict_len = len(label_dict)

    crf_cost, feature_out, word, mark, target = ner_net(word_dict_len, label_dict_len)    
    avg_cost = fluid.layers.mean(x=crf_cost)
    '''
    chunk_evaluator = fluid.evaluator.ChunkEvaluator(
	#name="ner_chunk",
        input=crf_decode,
        label=target,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((label_dict_len - 1) / 2.0)))
    '''
    #global_step = fluid.layers.create_global_var(
    #    shape=[1], value=0, dtype='float32', force_cpu=True, persistable=True)
    sgd_optimizer = fluid.optimizer.Momentum(
        momentum=0.0,
        learning_rate=2e-4,
        #regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.1)
	)
    sgd_optimizer.minimize(avg_cost)

    crf_decode = fluid.layers.crf_decoding(
        input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

    chunk_evaluator = fluid.evaluator.ChunkEvaluator(
        #name="ner_chunk",
        input=crf_decode,
        label=target,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((label_dict_len - 1) / 2.0)))

    #avg_cost = fluid.layers.mean(x=crf_cost)
    print type(crf_cost)
    #print type(avg_cost)
    print crf_cost.shape
    #print avg_cost.shape
    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_target = chunk_evaluator.metrics + chunk_evaluator.states
        inference_program = fluid.io.get_inference_program(test_target)
    
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.data_reader(train_data_file, word_dict, label_dict),
            buf_size=1000),
        batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.data_reader(test_data_file, word_dict, label_dict),
            buf_size=1000),
        batch_size=batch_size)

    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(
        feed_list=[
            word, mark, target
        ],
        place=place)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    embedding_name = 'emb'
    embedding_param = fluid.global_scope().find_var(embedding_name).get_tensor()
    embedding_param.set(word_vector_values, place)

    batch_id = 0
    for pass_id in xrange(num_passes):
        chunk_evaluator.reset(exe)
        for data in train_reader():
            cost, precision, recall, f1_score = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost] + chunk_evaluator.metrics)
	    if batch_id % 5 == 0:
                print("Pass " + str(pass_id) + ", Batch " + str(batch_id) + ", Cost " + str(cost) + ", Precision " + str(precision) + ", Recall " + str(recall) + ", F1_score" + str(f1_score))
            batch_id = batch_id + 1
	#pass_precision, pass_recall, pass_f1_score = chunk_evaluator.eval(exe)

        pass_precision, pass_recall, pass_f1_score = test(exe, chunk_evaluator, inference_program, train_reader, place)
	print("[TrainSet] pass_id:" + str(pass_id) + " pass_precision:" + str(pass_precision) + " pass_recall:" + str(pass_recall) + " pass_f1_score:" + str(pass_f1_score))
        pass_precision, pass_recall, pass_f1_score = test(exe, chunk_evaluator, inference_program, test_reader, place)
        print("[TestSet] pass_id:" + str(pass_id) + " pass_precision:" + str(pass_precision) + " pass_recall:" + str(pass_recall) + " pass_f1_score:" + str(pass_f1_score))


if __name__ == "__main__":
    main(
        train_data_file="full_conll03_dataset/train.txt",
        test_data_file="full_conll03_dataset/dev.txt",
        vocab_file="data/vocab.txt",
        target_file="data/target.txt",
        emb_file="data/wordVectors.txt",
        model_save_dir="models/",
	num_passes=1000)
