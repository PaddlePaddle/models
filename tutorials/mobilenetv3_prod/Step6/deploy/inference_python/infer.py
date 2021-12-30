import os
import argparse
import numpy as np
from PIL import Image

from preprocess_ops import ResizeImage, CenterCropImage, NormalizeImage, ToCHW, Compose

from paddle.inference import Config
from paddle.inference import PrecisionType
from paddle.inference import create_predictor

class InferenceEngine(object):
    """InferenceEngine
    
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
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
                x = np.random.rand(1, 3, 224, 224).astype("float32")
                self.input_tensor.copy_from_cpu(x)
                self.predictor.run()
                self.output_tensor.copy_to_cpu()
        return

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = Config(model_file_path, params_file_path)
        config.enable_memory_optim()
        if args.use_gpu:
            config.enable_use_gpu(100, 0)
            config.enable_tensorrt_engine(workspace_size=1 << 30,
                                      max_batch_size=10,
                                      min_subgraph_size=5,
                                      precision_mode=PrecisionType.Float32,
                                      use_static=False,
                                      use_calib_mode=False)
            config.set_trt_dynamic_shape_info(
                                      min_input_shape={"input": [1, 3, 1, 1]},
                                      max_input_shape={"input": [10, 3, 1200, 1200]},
                                      optim_input_shape={"input": [1, 3, 224, 224]})
        else:
            # If not specific mkldnn, you can set the blas thread.
            # The thread num should not be greater than the number of cores in the CPU.
            config.set_cpu_math_library_num_threads(4)
            config.enable_mkldnn()
        # creat predictor
        predictor = create_predictor(config)
        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_tensor, output_tensor

    def preprocess(self, img_path):
        """preprocess
        Preprocess to the input.
        Args:
            img_path: Image path.
        Returns: Input data after preprocess.
        """
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        img = self.transforms(img)
        img = np.expand_dims(img, axis=0)
        print(np.shape(img))
        return img

    def postprocess(self, x):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            x: Inference engine output.
        Returns: Output data after argmax.
        """
        x = x.flatten()
        class_id = x.argmax()
        prob = x[class_id]
        return class_id, prob

    def run(self, x):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        self.input_tensor.copy_from_cpu(x)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", type=int, default=1, help="Whether use gpu.")
    parser.add_argument("--warmup", default=1, type=int, help="warmup iter")
    parser.add_argument("--img-path", default="../../images/demo.jpg")
    parser.add_argument("--model-dir", default=None, help="inference model dir")
    parser.add_argument("--resize-size", default=256, type=int, help="resize_size")
    parser.add_argument("--crop-size", default=224, type=int, help="crop_szie")
    return parser.parse_args()

def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # preprocess
    img = inference_engine.preprocess(args.img_path)

    # run
    output = inference_engine.run(img)


    # postprocess
    class_id, prob = inference_engine.postprocess(output)

    # result
    print(f"image_name: {args.img_path}, class_id: {class_id}, prob: {prob}")
    return class_id, prob



if __name__ == '__main__':
    args = get_args()
    class_id, prob = infer_main(args)

