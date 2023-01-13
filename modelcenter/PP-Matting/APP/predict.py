import os
import codecs

import numpy as np
import cv2
import yaml
from paddle.inference import Config as PredictConfig
from paddle.inference import create_predictor

from download import download_file, uncompress

URL = 'https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmatting-hrnet_w18-human_512.zip'
SAVEPATH = './ppmatting-hrnet_w18-human_512.zip'


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
        # resize short edge to 512.
        h, w = img.shape[:2]
        short_edge = min(h, w)
        scale = 512 / short_edge
        h_resize = int(round(h * scale)) // 32 * 32
        w_resize = int(round(w * scale)) // 32 * 32
        img = cv2.resize(img, (w_resize, h_resize))
        img = (img / 255 - 0.5) / 0.5
        img = np.transpose(img, [2, 0, 1])[np.newaxis, :]
        return img

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
        input_handle['img'].copy_from_cpu(img_inputs)

        self.predictor.run()

        results = output_handle.copy_to_cpu()
        alpha = results.squeeze()
        alpha = cv2.resize(alpha, (ori_w, ori_h))
        alpha = (alpha * 255).astype('uint8')

        return alpha


def build_predictor():
    # Download inference model
    if not os.path.exists('./ppmatting-hrnet_w18-human_512'):
        download_file(url=URL, savepath=SAVEPATH)
        uncompress(SAVEPATH)
    cfg = os.path.join(os.path.splitext(SAVEPATH)[0], 'deploy.yaml')
    predictor = Predictor(cfg)
    return predictor
