import gradio as gr
import numpy as np

from paddleclas import PaddleClas
import cv2 as cv


clas = PaddleClas(model_name='PPLCNet_x0_25')

# UGC: Define the inference fn() for your models
def model_inference(image):
    result = clas.predict(image)
    return next(result)


def clear_all():
    return None, None


with gr.Blocks() as demo:
    gr.Markdown("Image Classification")

    with gr.Column(scale=1, min_width=100):

        img_in = gr.Image(
            value="https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.5/docs/images/inference_deployment/whl_demo.jpg",
            shape=(200, 200),
            label="Input").style(height=200)

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")

        prediction_out = gr.Textbox(label="Prediction")

    btn1.click(fn=clear_all, inputs=None, outputs=[img_in, prediction_out])
    btn2.click(fn=model_inference, inputs=img_in, outputs=[prediction_out])
    gr.Button.style(1)

demo.launch()
