import numpy as np

import cv2
import gradio as gr

from plsc.engine.inference import Predictor
from download import get_model_path, get_data_path

predictor = None


def model_inference(image):
    global predictor

    if predictor is None:

        model_path = "paddlecv://models/swin/v2.5/swin_base_patch4_window7_224_infer.pdmodel"
        params_path = "paddlecv://models/swin/v2.5/swin_base_patch4_window7_224_infer.pdiparams"
        label_path = "paddlecv://dataset/imagenet2012_labels.txt"
        infer_model = get_model_path(model_path)
        infer_params = get_model_path(params_path)

        def parse_labels(label_path):
            labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    if len(line) < 2:
                        continue
                    label = line.strip().split(',')[1]
                    labels.append(label)
            return np.array(labels)

        labels = parse_labels(get_data_path(label_path))

        def preprocess(img):
            img = cv2.resize(img, (224, 224))
            scale = 1.0 / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img * scale - mean) / std
            img = img[np.newaxis, :, :, :]
            img = img.transpose((0, 3, 1, 2))
            return {'x': img.astype('float32')}

        def postprocess(logits):
            def softmax(x, epsilon=1e-6):
                exp_x = np.exp(x)
                sfm = (exp_x + epsilon) / (np.sum(exp_x) + epsilon)
                return sfm

            pred = np.array(logits).squeeze()
            pred = softmax(pred)
            class_idx = pred.argsort()[::-1]
            class_idx_top5 = class_idx[:5]
            return class_idx_top5, pred[class_idx_top5], labels[class_idx_top5]

        predictor = Predictor(
            model_file=infer_model,
            params_file=infer_params,
            preprocess_fn=preprocess,
            postprocess_fn=postprocess)

    class_ids, scores, classes = predictor.predict(image)
    json_out = {
        "class_ids": class_ids.tolist(),
        "scores": scores.tolist(),
        "labels": classes.tolist()
    }
    return image, json_out


def clear_all():
    return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("Classification based on SwinTransformer")

    with gr.Column(scale=1, min_width=100):
        img_in = gr.Image(
            value="https://plsc.bj.bcebos.com/dataset/test_images/cat.jpg",
            label="Input").style(height=200)

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")

        img_out = gr.Image(label="Output").style(height=200)
        json_out = gr.JSON(label="jsonOutput")

    btn2.click(fn=model_inference, inputs=img_in, outputs=[img_out, json_out])
    btn1.click(fn=clear_all, inputs=None, outputs=[img_in, img_out, json_out])
    gr.Button.style(1)

demo.launch()
