import gradio as gr
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
from det_keypoint_unite_infer import def_keypoint


# UGC: Define the inference fn() for your models
def model_inference(input_date):
    if isinstance(input_date, str):
        if  os.path.splitext(input_date)[-1] not in ['.avi','.mp4']:
            return None,None
    pose,store_res = def_keypoint(input_date)
    json_out = {"result": store_res}
    return pose,json_out


def clear_all():
    return None, None,None


with gr.Blocks() as demo:
    gr.Markdown("Key Point Detection")

    with gr.Tabs():

        with gr.TabItem("image"):
           
            img_in =  gr.Image(value="https://paddledet.bj.bcebos.com/modelcenter/images/PP-TinyPose/000000568213.jpg",label="Input")
            img_out = gr.Image(label="Output")
            img_json_out = gr.JSON(label="jsonOutput")

            img_button1 = gr.Button("Submit")
            img_button2 = gr.Button("Clear")


        with gr.TabItem("video"):

            video_in = gr.Video(value="https://paddledet.bj.bcebos.com/modelcenter/images/PP-TinyPose/demo_PP-TinyPose.mp4",label="Input only support .mp4 or .avi")
            video_out = gr.Video(label="Output")
            video_json_out = gr.JSON(label="jsonOutput")

            video_button1 = gr.Button("Submit")
            video_button2 = gr.Button("Clear")

    img_button1.click(fn=model_inference, inputs=img_in, outputs=[img_out,img_json_out])
    img_button2.click(fn=clear_all, inputs=None, outputs=[img_in, img_out,img_json_out])

    video_button1.click(fn=model_inference, inputs=video_in, outputs=[video_out,video_json_out])
    video_button2.click(fn=clear_all, inputs=None, outputs=[video_in, video_out,video_json_out])


demo.launch()
