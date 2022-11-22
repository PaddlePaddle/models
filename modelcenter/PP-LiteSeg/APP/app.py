import codecs
import os
import sys
import time
import zipfile

import gradio as gr
import numpy as np
import cv2
import requests
import yaml
from paddle.inference import Config as PredictConfig
from paddle.inference import create_predictor
from PIL import Image as PILImage

lasttime = time.time()
FLUSH_INTERVAL = 0.1


def progress(str, end=False):
    global lasttime
    if end:
        str += "\n"
        lasttime = 0
    if time.time() - lasttime >= FLUSH_INTERVAL:
        sys.stdout.write("\r%s" % str)
        lasttime = time.time()
        sys.stdout.flush()


def _download_file(url, savepath, print_progress=True):
    if print_progress:
        print("Connecting to {}".format(url))
    r = requests.get(url, stream=True, timeout=15)
    total_length = r.headers.get('content-length')

    if total_length is None:
        with open(savepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    else:
        with open(savepath, 'wb') as f:
            dl = 0
            total_length = int(total_length)
            starttime = time.time()
            if print_progress:
                print("Downloading %s" % os.path.basename(savepath))
            for data in r.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                if print_progress:
                    done = int(50 * dl / total_length)
                    progress("[%-50s] %.2f%%" %
                             ('=' * done, float(100 * dl) / total_length))
        if print_progress:
            progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)


def uncompress(path):
    files = zipfile.ZipFile(path, 'r')
    filelist = files.namelist()
    rootpath = filelist[0]
    for file in filelist:
        files.extract(file, './')


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)
        self._dir = os.path.dirname(path)

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])


class Predictor:
    def __init__(self, cfg):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.cfg = DeployConfig(cfg)

        self._init_base_config()

        self._init_cpu_config()

        self.predictor = create_predictor(self.pred_cfg)

    def _init_base_config(self):
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        self.pred_cfg.disable_gpu()
        self.pred_cfg.set_cpu_math_library_num_threads(10)

    def _preprocess(self, img):
        # resize to (256, 144).
        img = (img / 255 - 0.5) / 0.5
        img = np.transpose(img, [2, 0, 1])[np.newaxis, :]
        return img

    def get_pseudo_color_map(self, pred, color_map=None):
        """
        Get the pseudo color image.
        Args:
            pred (numpy.ndarray): the origin predicted image.
            color_map (list, optional): the palette color map. Default: None,
                use paddleseg's default color map.
        Returns:
            (numpy.ndarray): the pseduo image.
        """
        pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
        if color_map is None:
            color_map = self.get_color_map_list(256)
        pred_mask.putpalette(color_map)
        return pred_mask


    def get_color_map_list(self, num_classes, custom_color=None):
        """
        Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
        Args:
            num_classes (int): Number of classes.
            custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.
        Returns:
            (list). The color map.
        """

        num_classes += 1
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = color_map[3:]

        if custom_color:
            color_map[:len(custom_color)] = custom_color
        return color_map

    def run(self, img):
        input_names = self.predictor.get_input_names()
        input_handle = {}

        for i in range(len(input_names)):
            input_handle[input_names[i]] = self.predictor.get_input_handle(
                input_names[i])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])

        img_inputs = img.astype('float32')
        ori_h, ori_w = img_inputs.shape[:2]
        img_inputs = self._preprocess(img=img_inputs)
        input_handle[input_names[0]].copy_from_cpu(img_inputs)

        self.predictor.run()

        results = output_handle.copy_to_cpu()
        result = results[0].squeeze().astype('uint8')
        result = cv2.resize(result, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        result = self.get_pseudo_color_map(result)

        # result = (result * 10).astype('uint8')

        return result


def model_inference(image):
    # Download inference model
    url = 'https://paddleseg.bj.bcebos.com/inference/pp_liteseg_infer_models/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k_inference_model.zip'
    savepath = './pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k_inference_model.zip'
    if not os.path.exists('./pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k_inference_model'):
        _download_file(url=url, savepath=savepath)
        uncompress(savepath)

    # Inference
    predictor = Predictor(cfg='./pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k_inference_model/deploy.yaml')
    alpha = predictor.run(image)

    return alpha


def clear_all():
    return None, None


with gr.Blocks() as demo:
    gr.Markdown("Segmentation")

    with gr.Column(scale=1, min_width=100):

        img_in = gr.Image(
            value="https://user-images.githubusercontent.com/48357642/201077761-3ebeda52-b15d-4913-b64c-0798d1f922a5.png",
            label="Input")

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")

        img_out = gr.Image(label="Output").style(height=200)

    btn2.click(fn=model_inference, inputs=img_in, outputs=[img_out])
    btn1.click(fn=clear_all, inputs=None, outputs=[img_in, img_out])
    gr.Button.style(1)

demo.launch()
