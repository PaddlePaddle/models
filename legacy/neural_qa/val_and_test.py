import os
import sys
import argparse
import time
import traceback
import subprocess
import re

import utils
import infer
import config
from utils import logger


def load_existing_results(eval_result_file):
    evals = {}
    with utils.open_file(eval_result_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            pos = line.find(" ")
            pass_id, ret = int(line[len("Pass="):pos]), line[pos + 1:]
            evals[pass_id] = ret
    return evals


__PATTERN_CHUNK_F1 = re.compile("chunk_f1=(\d+(\.\d+)?)")


def find_best_pass(evals):
    results = []
    for pass_id, eval_ret in evals.iteritems():
        chunk_f1 = float(__PATTERN_CHUNK_F1.search(eval_ret).group(1))
        results.append((pass_id, chunk_f1))

    results.sort(key=lambda item: (-item[1], item[0]))
    return results[0][0]


def eval_one_pass(infer_obj, conf, model_path, data_path, eval_script):
    if not os.path.exists("tmp"): os.makedirs("tmp")
    # model file is not ready
    if not os.path.exists(model_path): return False

    output_path = os.path.join("tmp", "%s_%s.txt.gz" % (
        os.path.basename(model_path), os.path.basename(data_path)))
    with utils.open_file(output_path, "w") as output:
        try:
            infer_obj.infer(model_path, data_path, output)
        except Exception as ex:
            traceback.print_exc()
            return None

    cmd = [
        "python", eval_script, output_path, data_path, "--fuzzy", "--schema",
        conf.label_schema
    ]
    logger.info("cmd: %s" % " ".join(cmd))
    eval_ret = subprocess.check_output(cmd)
    if "chunk_f1" not in eval_ret:
        raise ValueError("Unknown error in cmd \"%s\"" % " ".join(cmd))

    return eval_ret


def run_eval(infer_obj,
             conf,
             model_dir,
             input_path,
             eval_script,
             log_file,
             start_pass_id,
             end_pass_id,
             force_rerun=False):
    if not force_rerun and os.path.exists(log_file):
        evals = load_existing_results(log_file)
    else:
        evals = {}
    with utils.open_file(log_file, "w") as log:
        for i in xrange(start_pass_id, end_pass_id + 1):
            if i in evals:
                eval_ret = evals[i]
            else:
                pass_id = "%05d" % i
                model_path = os.path.join(model_dir,
                                          "params_pass_%s.tar.gz" % pass_id)
                logger.info("Waiting for model %s ..." % model_path)
                while True:
                    eval_ret = eval_one_pass(infer_obj, conf, model_path,
                                             input_path, eval_script)
                    if eval_ret:
                        evals[i] = eval_ret
                        break

                    # wait for one minute and rerun
                    time.sleep(60)
            print >> log, "Pass=%d %s" % (i, eval_ret.rstrip())
            log.flush()
    return evals


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("data_type", choices=["ann", "ir"], default="ann")
    parser.add_argument(
        "--val_eval_output", help="validation set evaluation result file")
    parser.add_argument(
        "--tst_eval_output", help="test set evaluation result file")
    parser.add_argument("--start_pass_id", type=int, default=0)
    parser.add_argument(
        "--end_pass_id", type=int, default=24, help="this pass is included")
    parser.add_argument("--force_rerun", action="store_true")
    return parser.parse_args()


__eval_scripts = {
    "ann": "data/evaluation/evaluate-tagging-result.py",
    "ir": "data/evaluation/evaluate-voting-result.py",
}

__val_data = {
    "ann": "./data/data/validation.ann.json.gz",
    "ir": "./data/data/validation.ir.json.gz",
}

__tst_data = {
    "ann": "./data/data/test.ann.json.gz",
    "ir": "./data/data/test.ir.json.gz",
}


def main(args):
    conf = config.InferConfig()
    conf.vocab = utils.load_dict(conf.word_dict_path)
    logger.info("length of word dictionary is : %d." % len(conf.vocab))

    if args.val_eval_output:
        val_eval_output = args.val_eval_output
    else:
        val_eval_output = "eval.val.%s.txt" % args.data_type

    if args.tst_eval_output:
        tst_eval_output = args.tst_eval_output
    else:
        tst_eval_output = "eval.tst.%s.txt" % args.data_type

    eval_script = __eval_scripts[args.data_type]
    val_data_file = __val_data[args.data_type]
    tst_data_file = __tst_data[args.data_type]

    infer_obj = infer.Infer(conf)
    val_evals = run_eval(
        infer_obj,
        conf,
        args.model_dir,
        val_data_file,
        eval_script,
        val_eval_output,
        args.start_pass_id,
        args.end_pass_id,
        force_rerun=args.force_rerun)

    best_pass_id = find_best_pass(val_evals)

    tst_evals = run_eval(
        infer_obj,
        conf,
        args.model_dir,
        tst_data_file,
        eval_script,
        tst_eval_output,
        start_pass_id=best_pass_id,
        end_pass_id=best_pass_id,
        force_rerun=args.force_rerun)

    logger.info("Best Pass=%d" % best_pass_id)
    logger.info("Validation: %s" % val_evals[best_pass_id])
    logger.info("Test      : %s" % tst_evals[best_pass_id])


if __name__ == "__main__":
    main(parse_cmd())
