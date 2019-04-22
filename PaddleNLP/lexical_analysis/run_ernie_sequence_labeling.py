"""
Sentiment Classification Task
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import argparse
import numpy as np
import multiprocessing
import sys

import paddle
import paddle.fluid as fluid
from collections import namedtuple

sys.path.append("..")
print(sys.path)
from preprocess.ernie import task_reader

from models.representation.ernie import ErnieConfig
from models.representation.ernie import ernie_encoder
#from models.representation.ernie import ernie_pyreader
from models.sequence_labeling import nets
import utils


# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = utils.ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("ernie_config_path", str, "../LARK/ERNIE/config/ernie_config.json",
        "Path to the json file for ernie model config.")
model_g.add_arg("lac_config_path", str, None, "Path to the json file for LAC model config.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("checkpoints", str, None, "Path to save checkpoints")
model_g.add_arg("init_pretraining_params", str, "pretrained/params/",
                "Init pre-training params which preforms fine-tuning from. If the "
                 "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")

train_g = utils.ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 10, "Number of epoches for training.")
train_g.add_arg("save_steps", int, 10000, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")
train_g.add_arg("lr", float, 0.001, "The Learning rate value for training.")
train_g.add_arg("crf_learning_rate", float, 0.2,
    "The real learning rate of the embedding layer will be (crf_learning_rate * base_learning_rate).")
train_g.add_arg("init_bound", float, 0.1, "init bound for initialization.")

log_g = utils.ArgumentGroup(parser, "logging", "logging related")
log_g.add_arg("skip_steps", int, 1, "The steps interval to print loss.")

data_g = utils.ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("vocab_path", str, "../LARK/ERNIE/config/vocab.txt", "Vocabulary path.")
data_g.add_arg("batch_size", int, 3, "Total examples' number in batch for training.")
data_g.add_arg("random_seed", int, 0, "Random seed.")
data_g.add_arg("num_labels", int, 57, "label number")
data_g.add_arg("max_seq_len", int, 512, "Number of words of the longest seqence.")
data_g.add_arg("train_set", str, "./data/train.tsv", "Path to train data.")
data_g.add_arg("test_set", str, "./data/test.tsv", "Path to test data.")
data_g.add_arg("infer_set", str, "./data/test.tsv", "Path to infer data.")
data_g.add_arg("label_map_config", str, "./conf/label_map.json", "label_map_path.")
data_g.add_arg("do_lower_case", bool, True,
        "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

run_type_g = utils.ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_test", bool, True, "Whether to perform testing.")
run_type_g.add_arg("do_infer", bool, True, "Whether to perform inference.")

args = parser.parse_args()
# yapf: enable.

def ernie_pyreader(args, pyreader_name):
    """define standard ernie pyreader"""
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, 1]],
        dtypes=['int64', 'int64', 'int64', 'float32', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, input_mask, padded_labels, seq_lens) = fluid.layers.read_file(pyreader)

    words = fluid.layers.sequence_unpad(src_ids, seq_lens)
    labels = fluid.layers.sequence_unpad(padded_labels, seq_lens)

    ernie_inputs = {
        "src_ids": src_ids,
        "sent_ids": sent_ids,
        "pos_ids": pos_ids,
        "input_mask": input_mask,
        "seq_lens": seq_lens
    }
    return pyreader, ernie_inputs, words, labels


def create_model(args,
                 embeddings,
                 labels,
                 is_prediction=False):

    """
    Create Model for LAC based on ERNIE encoder
    """
    # sentence_embeddings = embeddings["sentence_embeddings"]
    token_embeddings = embeddings["token_embeddings"]

    emission = fluid.layers.fc(
        size=args.num_labels,
        input=token_embeddings,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-args.init_bound, high=args.init_bound),
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=1e-4)))

    crf_cost = fluid.layers.linear_chain_crf(
        input=emission,
        label=labels,
        param_attr=fluid.ParamAttr(
            name='crfw',
            learning_rate=args.crf_learning_rate))
    loss = fluid.layers.mean(x=crf_cost)

    crf_decode = fluid.layers.crf_decoding(
        input=emission, param_attr=fluid.ParamAttr(name='crfw'))

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
         input=crf_decode,
         label=labels,
         chunk_scheme="IOB",
         num_chunk_types=int(math.ceil((args.num_labels - 1) / 2.0)))
    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()

    ret = {
        "loss":loss,
        "crf_decode":crf_decode,
        "chunk_evaluator":chunk_evaluator,
        "num_infer_chunks":num_infer_chunks,
        "num_label_chunks":num_label_chunks,
        "num_correct_chunks":num_correct_chunks
    }

    return ret


def evaluate(exe, test_program, test_pyreader, test_ret):
    """
    Evaluation Function
    """
    test_pyreader.start()
    test_ret["chunk_evaluator"].reset()
    total_loss, precision, recall, f1 = [], [], [], []
    start_time = time.time()
    while True:
        try:
            loss, nums_infer, nums_label, nums_correct = exe.run(
                test_program,
                fetch_list=[
                    test_ret["loss"],
                    test_ret["num_infer_chunks"],
                    test_ret["num_label_chunks"],
                    test_ret["num_correct_chunks"],
                ],
            )
            total_loss.append(loss)

            test_ret["chunk_evaluator"].update(nums_infer, nums_label, nums_correct)
            p, r, f = test_ret["chunk_evaluator"].eval()
            precision.append(p)
            recall.append(r)
            f1.append(f)

        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    end_time = time.time()
    print("\t[test] loss: %.5f, P: %.5f, R: %.5f, F1: %.5f, elapsed time: %.3f s"
        % (np.mean(total_loss), np.mean(precision), np.mean(recall), np.mean(f1), end_time - start_time))


def main(args):
    """
    Main Function
    """
    args = parser.parse_args()
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    reader = task_reader.SequenceLabelReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=False,
        random_seed=args.random_seed)

    if not (args.do_train or args.do_test or args.do_infer):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        num_train_examples = reader.get_num_examples(args.train_set)
        max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count
        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                # create ernie_pyreader
                train_pyreader, ernie_inputs, words, labels = ernie_pyreader(args, pyreader_name='train_reader')
                train_pyreader.decorate_tensor_provider(
                    reader.data_generator(
                        args.train_set, args.batch_size, args.epoch, shuffle=True, phase="train"
                    )
                )
                # get ernie_embeddings
                embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)
                # user defined model based on ernie embeddings
                train_ret = create_model(args, embeddings, labels=labels, is_prediction=False)

                optimizer = fluid.optimizer.Adam(learning_rate=args.lr)
                fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))
                optimizer.minimize(train_ret["loss"])

        lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
            program=train_program, batch_size=args.batch_size)
        print("Theoretical memory usage in training: %.3f - %.3f %s" %
            (lower_mem, upper_mem, unit))

    if args.do_test:
        test_program = fluid.Program()
        with fluid.program_guard(test_program, startup_prog):
            with fluid.unique_name.guard():
                # create ernie_pyreader
                test_pyreader, ernie_inputs, words, labels = ernie_pyreader(args, pyreader_name='test_reader')
                test_pyreader.decorate_tensor_provider(
                    reader.data_generator(
                        args.test_set, args.batch_size, phase='test', epoch=1, shuffle=False
                    )
                )
                # get ernie_embeddings
                embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)
                # user defined model based on ernie embeddings
                test_ret = create_model(args, embeddings, labels=labels, is_prediction=False)

        test_program = test_program.clone(for_test=True)

    if args.do_infer:
        infer_program = fluid.Program()
        with fluid.program_guard(infer_program, startup_prog):
            with fluid.unique_name.guard():
                # create ernie_pyreader
                infer_pyreader, ernie_inputs, words, labels = ernie_pyreader(args, pyreader_name='infer_reader')
                infer_pyreader.decorate_tensor_provider(
                    reader.data_generator(
                        args.infer_set, args.batch_size, phase='infer', epoch=1, shuffle=False
                    )
                )
                # get ernie_embeddings
                embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)
                # user defined model based on ernie embeddings
                infer_ret = create_model(args, embeddings, labels=labels, is_prediction=True)
                infer_ret["words"] = words

        infer_program = infer_program.clone(for_test=True)

    exe.run(startup_prog)

    # load checkpoints
    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            print("WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                    "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            utils.init_checkpoint(exe, args.init_checkpoint, startup_prog)
        elif args.init_pretraining_params:
            utils.init_pretraining_params(exe, args.init_pretraining_params, startup_prog)
    elif args.do_test or args.do_infer:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if only doing test or infer!")
        utils.init_checkpoint(exe, args.init_checkpoint, startup_prog)

    if args.do_train:
        train_pyreader.start()
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []
        while True:
            try:
                steps += 1
                if steps % args.skip_steps == 0:
                    fetch_list = [
                        train_ret["loss"],
                        train_ret["num_infer_chunks"],
                        train_ret["num_label_chunks"],
                        train_ret["num_correct_chunks"],
                    ]
                else:
                    fetch_list = []

                start_time = time.time()
                outputs = exe.run(program=train_program, fetch_list=fetch_list)
                end_time = time.time()
                if steps % args.skip_steps == 0:
                    loss, nums_infer, nums_label, nums_correct = outputs
                    train_ret["chunk_evaluator"].reset()
                    train_ret["chunk_evaluator"].update(nums_infer, nums_label, nums_correct)
                    precision, recall, f1_score = train_ret["chunk_evaluator"].eval()
                    print("[train] batch_id = %d, loss = %.5f, P: %.5f, R: %.5f, F1: %.5f, elapsed time %.5f, "
                            "pyreader queue_size: %d " % (steps, loss, precision, recall, f1_score,
                            end_time - start_time, train_pyreader.queue.size()))

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                    print("\tsaving model as %s" % (save_path))
                    fluid.io.save_persistables(exe, save_path, train_program)

                if steps % args.validation_steps == 0:
                    # evaluate test set
                    if args.do_test:
                        evaluate(exe, test_program, test_pyreader, test_ret)

            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # final eval on test set
    if args.do_test:
        evaluate(exe, test_program, test_pyreader, test_ret)

    if args.do_infer:
        # create dict
        id2word_dict = dict([(str(word_id), word) for word, word_id in reader.vocab.items()])
        id2label_dict = dict([(str(label_id), label) for label, label_id in reader.label_map.items()])
        Dataset = namedtuple("Dataset", ["id2word_dict", "id2label_dict"])
        dataset = Dataset(id2word_dict, id2label_dict)

        infer_pyreader.start()
        while True:
            try:
                (words, crf_decode) = exe.run(infer_program,
                        fetch_list=[infer_ret["words"], infer_ret["crf_decode"]],
                        return_numpy=False)
                # User should notice that words had been clipped if long than args.max_seq_len
                results = utils.parse_result(words, crf_decode, dataset)
                for result in results:
                    print(result)
            except fluid.core.EOFException:
                infer_pyreader.reset()
                break


if __name__ == "__main__":
    utils.print_arguments(args)
    main(args)
