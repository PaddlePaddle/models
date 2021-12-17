import os
import paddle
from paddle import inference
import numpy as np
from PIL import Image

from reprod_log import ReprodLogger
from preprocess_ops import ResizeImage, CenterCropImage, NormalizeImage, ToCHW, Compose


class InferenceEngine(object):
    def __init__(self, args):
        super().__init__()
        pass

    def load_predictor(self, model_file_path, params_file_path):
        """
        initialize the inference engine
        """
        pass

    def preprocess(self, img_path):
        # preprocess for data
        pass

    def postprocess(self, x):
        # postprocess for the inference engine output
        pass

    def run(self, x):
        # run using the infer
        pass


def get_args(add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle", add_help=add_help)

    args = parser.parse_args()
    return args


def infer_main(args):
    # init inference engine
    inference_engine = InferenceEngine(args)

    # init benchmark log
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="example",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img = inference_engine.preprocess(args.img_path)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(img)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    class_id, prob = inference_engine.postprocess(output)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    return class_id, prob


if __name__ == "__main__":
    args = get_args()
    infer_main(args)