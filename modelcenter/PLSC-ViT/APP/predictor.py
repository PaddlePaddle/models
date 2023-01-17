import os
import cv2
import numpy as np
import paddle
from download import get_model_path, get_data_path


class Predictor(object):
    def __init__(self,
                 model_type="paddle",
                 model_path=None,
                 params_path=None,
                 label_path=None):
        '''
        model_path: str, http url
        params_path: str, http url, could be downloaded
        '''
        assert model_type in ["paddle"]
        assert model_path is not None and os.path.splitext(model_path)[
            1] == '.pdmodel'
        assert params_path is not None and os.path.splitext(params_path)[
            1] == '.pdiparams'

        import paddle.inference as paddle_infer
        infer_model = get_model_path(model_path)
        infer_params = get_model_path(params_path)
        config = paddle_infer.Config(infer_model, infer_params)
        self.predictor = paddle_infer.create_predictor(config)
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        self.labels = self.parse_labes(get_data_path(label_path))
        self.model_type = model_type

    def predict(self, img):

        if self.preprocess is not None:
            inputs = self.preprocess(img)
        else:
            inputs = img
        for input_name in self.input_names:
            input_tensor = self.predictor.get_input_handle(input_name)
            input_tensor.copy_from_cpu(inputs[input_name])
        self.predictor.run()
        outputs = []
        for output_idx in range(len(self.output_names)):
            output_tensor = self.predictor.get_output_handle(self.output_names[
                output_idx])
            outputs.append(output_tensor.copy_to_cpu())
        if self.postprocess is not None:
            output_data = self.postprocess(outputs)
        else:
            output_data = outputs

        return output_data

    def preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        scale = 1.0 / 255.0
        mean = 0.5
        std = 0.5
        img = (img.astype('float32') * scale - mean) / std
        img = img[np.newaxis, :, :, :]
        img = img.transpose((0, 3, 1, 2))
        return {'x': img}

    @staticmethod
    def parse_labes(label_path):
        with open(label_path, 'r') as f:
            labels = []
            for line in f:
                if len(line) < 2:
                    continue
                label = line.strip().split(',')[1].strip()
                labels.append(label)
        return labels

    @staticmethod
    def softmax(x, epsilon=1e-6):
        exp_x = np.exp(x)
        sfm = (exp_x + epsilon) / (np.sum(exp_x) + epsilon)
        return sfm

    def postprocess(self, logits):
        pred = np.array(logits).squeeze()
        pred = self.softmax(pred)
        class_idx = pred.argsort()[::-1]
        return class_idx[:5], pred[class_idx[:5]], np.array(self.labels)[
            class_idx[:5]]
