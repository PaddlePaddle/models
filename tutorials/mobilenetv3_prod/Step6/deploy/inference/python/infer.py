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
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensor, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

        # build transforms
        self.transforms = Compose([
            ResizeImage(args.resize_size), CenterCropImage(args.crop_size),
            NormalizeImage(), ToCHW()
        ])

        # wamrup
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                print(idx)
                x = np.random.rand(1, 3, self.args.crop_size,
                                   self.args.crop_size).astype("float32")
                self.input_tensor.copy_from_cpu(x)
                self.predictor.run()
                self.output_tensor.copy_to_cpu()
        return

    def load_predictor(self, model_file_path, params_file_path):
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()
            if args.use_mkldnn:
                config.enable_mkldnn()
                config.set_cpu_math_library_num_threads(args.cpu_threads)

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_tensor, output_tensor

    def preprocess(self, img_path):
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        img = self.transforms(img)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, x):
        x = x.flatten()
        class_id = x.argmax()
        prob = x[class_id]
        return class_id, prob

    def run(self, x):
        # inference using inference engine
        self.input_tensor.copy_from_cpu(x)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output


def get_args(add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)

    parser.add_argument(
        "--model-dir", default=None, help="inference model dir")
    parser.add_argument(
        "--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument(
        "--use-mkldnn", default=False, type=str2bool, help="use_mkldnn")
    parser.add_argument(
        "--min-subgraph-size", default=15, type=int, help="min_subgraph_size")
    parser.add_argument(
        "--max-batch-size", default=16, type=int, help="max_batch_size")
    parser.add_argument(
        "--cpu-threads", default=10, type=int, help="cpu-threads")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")

    parser.add_argument(
        "--resize-size", default=256, type=int, help="resize_size")
    parser.add_argument("--crop-size", default=224, type=int, help="crop_szie")
    parser.add_argument("--img-path", default="./images/demo.jpg")

    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")

    args = parser.parse_args()
    return args


def infer_main(args):
    # init inference engine
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="classification",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

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

    print(f"image_name: {args.img_path}, class_id: {class_id}, prob: {prob}")
    return class_id, prob


if __name__ == "__main__":
    args = get_args()
    class_id, prob = infer_main(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("class_id", np.array([class_id]))
    reprod_logger.add("prob", np.array([prob]))
    reprod_logger.save("output_inference_engine.npy")
