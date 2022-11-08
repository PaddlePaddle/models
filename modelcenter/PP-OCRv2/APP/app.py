import gradio as gr
import base64
from io import BytesIO
from PIL import Image

from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR(ocr_version='PP-OCRv2', use_angle_cls=True, lang="ch")


def image_to_base64(image):
    # 输入为PIL读取的图片，输出为base64格式
    byte_data = BytesIO()  # 创建一个字节流管道
    image.save(byte_data, format="JPEG")  # 将图片数据存入字节流管道
    byte_data = byte_data.getvalue()  # 从字节流管道中获取二进制
    base64_str = base64.b64encode(byte_data).decode("ascii")  # 二进制转base64
    return base64_str


# UGC: Define the inference fn() for your models
def model_inference(image):
    result = ocr.ocr(image, cls=True)

    # 显示结果
    result = result[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts=None, scores=None)
    im_show = Image.fromarray(im_show)

    res = []
    for i in range(len(boxes)):
        res.append(dict(boxes=boxes[i], txt=txts[i], score=scores[i]))
    json_out = {"base64": image_to_base64(im_show), "result": res}
    return im_show, json_out


def clear_all():
    return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("PP-OCRv2")

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
