import gradio as gr
import base64
from io import BytesIO
from PIL import Image

from paddleocr import PPStructure

table_engine = PPStructure(layout=False, show_log=True)


def image_to_base64(image):
    # 输入为PIL读取的图片，输出为base64格式
    byte_data = BytesIO()  # 创建一个字节流管道
    image.save(byte_data, format="JPEG")  # 将图片数据存入字节流管道
    byte_data = byte_data.getvalue()  # 从字节流管道中获取二进制
    base64_str = base64.b64encode(byte_data).decode("ascii")  # 二进制转base64
    return base64_str


# UGC: Define the inference fn() for your models
def model_inference(image):
    result = table_engine(image)
    res = result[0]['res']['html']
    json_out = {"result": res}
    return res, json_out


def clear_all():
    return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("PP-StructureV2")

    with gr.Column(scale=1, min_width=100):
        img_in = gr.Image(
            value="https://user-images.githubusercontent.com/12406017/200574299-32537341-c329-42a5-ae41-35ee4bd43f2f.png",
            label="Input")

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")

        html_out = gr.HTML(label="Output")
        json_out = gr.JSON(label="jsonOutput")

    btn2.click(fn=model_inference, inputs=img_in, outputs=[html_out, json_out])
    btn1.click(fn=clear_all, inputs=None, outputs=[img_in, html_out, json_out])
    gr.Button.style(1)

demo.launch()
