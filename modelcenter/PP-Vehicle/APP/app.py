import gradio as gr
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
from pipeline.pipeline import pp_vehicls


# UGC: Define the inference fn() for your models
def model_inference(input_date, avtivity_list):

    if isinstance(input_date, str):
        if  os.path.splitext(input_date)[-1] not in ['.avi','.mp4']:
            return None
            
    result = pp_vehicls(input_date, avtivity_list)

    return result


def clear_all():
    return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("PP-Vehicle Pipeline")

    with gr.Tabs():

        with gr.TabItem("image"):

            img_in = gr.Image(value="https://paddledet.bj.bcebos.com/modelcenter/images/PP-Vehicle/demo_vehicle.jpg",label="Input")
            img_out = gr.Image(label="Output")

            img_avtivity_list = gr.CheckboxGroup(
                ["VEHICLE_PLATE", "VEHICLE_ATTR"])
            img_button1 = gr.Button("Submit")
            img_button2 = gr.Button("Clear")

        with gr.TabItem("video"):

            video_in = gr.Video(value="https://paddledet.bj.bcebos.com/modelcenter/images/PP-Vehicle/demo_vehicle.mp4",label="Input only support .mp4 or .avi")
            video_out = gr.Video(label="Output")

            video_avtivity_list = gr.CheckboxGroup(
                ["MOT", "VEHICLE_PLATE", "VEHICLE_ATTR"])
            video_button1 = gr.Button("Submit")
            video_button2 = gr.Button("Clear")

    img_button1.click(
        fn=model_inference,
        inputs=[img_in, img_avtivity_list],
        outputs=img_out)
    img_button2.click(
        fn=clear_all,
        inputs=None,
        outputs=[img_in, img_out, img_avtivity_list])

    video_button1.click(
        fn=model_inference,
        inputs=[video_in, video_avtivity_list],
        outputs=video_out)
    video_button2.click(
        fn=clear_all,
        inputs=None,
        outputs=[video_in, video_out, video_avtivity_list])

demo.launch()
