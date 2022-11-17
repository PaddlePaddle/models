import gradio as gr
import numpy as np
import os
from src.detection import Detector


# UGC: Define the inference fn() for your models
def model_inference(image):
    image, json_out = Detector('PP-YOLOv2')(image)
    return image, json_out


def clear_all():
    return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("Objective Detection")

    with gr.Column(scale=1, min_width=100):

        img_in = gr.Image(label="Input").style(height=200)

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")

        img_out = gr.Image(label="Output").style(height=200)
        json_out = gr.JSON(label="jsonOutput")

    btn2.click(fn=model_inference, inputs=img_in, outputs=[img_out, json_out])
    btn1.click(fn=clear_all, inputs=None, outputs=[img_in, img_out, json_out])
    gr.Button.style(1)

demo.launch()
