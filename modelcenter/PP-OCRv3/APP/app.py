import gradio as gr
import base64
from io import BytesIO
from PIL import Image

from paddlecv import PaddleCV
ocr = PaddleCV(task_name="PP-OCRv3")


def image_to_base64(image):
    # 输入为PIL读取的图片，输出为base64格式
    byte_data = BytesIO()  # 创建一个字节流管道
    image.save(byte_data, format="JPEG")  # 将图片数据存入字节流管道
    byte_data = byte_data.getvalue()  # 从字节流管道中获取二进制
    base64_str = base64.b64encode(byte_data).decode("ascii")  # 二进制转base64
    return base64_str


# UGC: Define the inference fn() for your models
def model_inference(image):
    result = ocr(image)[0]

    im_show = Image.open('output/tmp.jpg')
    res = []
    for i in range(len(result['dt_polys'])):
        res.append(
            dict(
                boxes=result['dt_polys'][i],
                txt=result['rec_text'][i],
                score=result['rec_score'][i]))
    json_out = {"base64": image_to_base64(im_show), "result": res}
    return im_show, json_out


def clear_all():
    return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("PP-OCRv3")

    with gr.Column(scale=1, min_width=100):
        img_in = gr.Image(
            value="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/dygraph/doc/imgs/11.jpg",
            label="Input")

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")
        img_out = gr.Image(label="Output").style(height=400)
        json_out = gr.JSON(label="jsonOutput")

    btn2.click(fn=model_inference, inputs=img_in, outputs=[img_out, json_out])
    btn1.click(fn=clear_all, inputs=None, outputs=[img_in, img_out, json_out])
    gr.Button.style(1)

demo.launch()
