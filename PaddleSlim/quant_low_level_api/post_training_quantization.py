import sys
import os
import six
import numpy as np
import argparse
import paddle.fluid as fluid
sys.path.append('..')
import reader
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir", type=str, default="", help="path/to/fp32_model_params")
parser.add_argument(
    "--data_path", type=str, default="/dataset/ILSVRC2012/", help="")
parser.add_argument("--save_model_path", type=str, default="")
parser.add_argument(
    "--model_filename",
    type=str,
    default=None,
    help="The name of file to load the inference program, If it is None, the default filename __model__ will be used"
)
parser.add_argument(
    "--params_filename",
    type=str,
    default=None,
    help="The name of file to load all parameters, If parameters were saved in separate files, set it as None"
)
parser.add_argument(
    "--algo",
    type=str,
    default="KL",
    help="use KL or direct method to quantize the activation tensor, set it as KL or direct"
)
parser.add_argument("--is_full_quantize", type=str, default="False", help="")
parser.add_argument("--batch_size", type=int, default=10, help="")
parser.add_argument("--batch_nums", type=int, default=10, help="")
parser.add_argument("--use_gpu", type=str, default="False", help="")
args = parser.parse_args()

print("-------------------args----------------------")
for arg, value in sorted(six.iteritems(vars(args))):
    print("%s: %s" % (arg, value))
print("---------------------------------------------")

place = fluid.CUDAPlace(0) if args.use_gpu == "True" else fluid.CPUPlace()
exe = fluid.Executor(place)
sample_generator = reader.val(data_dir=args.data_path)

ptq = PostTrainingQuantization(
    executor=exe,
    sample_generator=sample_generator,
    model_dir=args.model_dir,
    model_filename=args.model_filename,
    params_filename=args.params_filename,
    batch_size=args.batch_size,
    batch_nums=args.batch_nums,
    algo=args.algo,
    is_full_quantize=args.is_full_quantize == "True")
quantized_program = ptq.quantize()
ptq.save_quantized_model(args.save_model_path)

print("post training quantization finish.\n")
