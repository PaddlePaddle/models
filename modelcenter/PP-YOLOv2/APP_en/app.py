import gradio as gr
import numpy as np

import cv2 as cv


def model_inference(image):
    json_out = {
        "base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAg...",
        "result": "123456"
    }
    return image, json_out


def clear_all():
    return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("Objective Detection")

    with gr.Column(scale=1, min_width=100):

        img_in = gr.Image(
            value="https://i.picsum.photos/id/867/600/600.jpg?hmac=qE7QFJwLmlE_WKI7zMH6SgH5iY5fx8ec6ZJQBwKRT44",
            shape=(200, 200),
            label="Input").style(height=200)

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")

        img_out = gr.Image(shape=(200, 200), label="Output").style(height=200)
        json_out = gr.JSON(label="jsonOutput")

    btn2.click(fn=model_inference, inputs=img_in, outputs=[img_out, json_out])
    btn1.click(fn=clear_all, inputs=None, outputs=[img_in, img_out, json_out])
    gr.Button.style(1)

demo.launch()
