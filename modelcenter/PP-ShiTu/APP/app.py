import base64
import os
from io import BytesIO
from typing import Any, Dict, List, Union

import gradio as gr
import numpy as np
import requests
from paddleclas import PaddleClas
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def download_with_progressbar(url: str, save_path: str):
    """Download file from given url and decompress it

    Args:
        url (str): url
        save_path (str): path for saving downloaded file

    Raises:
        Exception: exception
    """
    print(f"Auto downloading {url} to {save_path}")
    if os.path.exists(save_path):
        print("File already exist, skip...")
    else:
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes or not os.path.isfile(
                save_path):
            raise Exception(
                f"Something went wrong while downloading file from {url}")
        print("Finished downloading")
        print(f"Try decompression at {save_path}")
        os.system(f"tar -xf {save_path}")
        print(f"Finished decompression at {save_path}")


def image_to_base64(image: Image.Image) -> str:
    """encode Pillow image to base64 string

    Args:
        image (Image.Image): image to be encoded

    Returns:
        str: encoded string
    """
    byte_data = BytesIO()  # 创建一个字节流管道
    image.save(byte_data, format="JPEG")  # 将图片数据存入字节流管道
    byte_data = byte_data.getvalue()  # 从字节流管道中获取二进制
    base64_str = base64.b64encode(byte_data).decode("ascii")  # 二进制转base64
    return base64_str


# UGC: Define the inference fn() for your models
def model_inference(image) -> tuple:
    """send given image to inference model and get result from output

    Args:
        image (gr.Image): _description_

    Returns:
        tuple: (image to display, result in json format)
    """
    results = clas_engine.predict(image, print_pred=True, predict_type="shitu")

    # bs = 1, fetch the first result
    results = list(results)[0]

    image_draw_box = draw_bbox_results(image, results)

    im_show = Image.fromarray(image_draw_box)

    json_out = {"base64": image_to_base64(im_show), "result": str(results)}
    return im_show, json_out


def draw_bbox_results(image: Union[np.ndarray, Image.Image],
                      results: List[Dict[str, Any]]) -> np.ndarray:
    """draw bounding box(es)

    Args:
        image (Union[np.ndarray, Image.Image]): image to be drawn
        results (List[Dict[str, Any]]): information for drawing bounding box

    Returns:
        np.ndarray: drawn image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font_size = 18
    font = ImageFont.truetype("./simfang.ttf", font_size, encoding="utf-8")

    color = (0, 102, 255)

    for result in results:
        # empty results
        if result["rec_docs"] is None:
            continue

        xmin, ymin, xmax, ymax = result["bbox"]
        text = "{}, {:.2f}".format(result["rec_docs"], result["rec_scores"])
        th = font_size
        tw = font.getsize(text)[0]
        start_y = max(0, ymin - th)

        draw.rectangle(
            [(xmin + 1, start_y), (xmin + tw + 1, start_y + th)], fill=color)

        draw.text((xmin + 1, start_y), text, fill=(255, 255, 255), font=font)

        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)], outline=(255, 0, 0), width=2)

    return np.array(image)


def clear_all():
    return None, None, None


# download drink_dataset_v2.0.tar
dataset_url = "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar"
download_with_progressbar(dataset_url,
                          os.path.join("./", dataset_url.split("/")[-1]))

clas_engine = PaddleClas(model_name="PP-ShiTuV2", use_gpu=False)

with gr.Blocks() as demo:
    gr.Markdown("PP-ShiTuV2")

    with gr.Column(scale=1, min_width=100):
        img_in = gr.Image(
            value="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/images/recognition/drink_data_demo/test_images/100.jpeg?raw=true",
            label="Input")

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")
        img_out = gr.Image(label="Output").style(height=400)
        json_out = gr.JSON(label="jsonOutput")

    btn2.click(fn=model_inference, inputs=img_in, outputs=[img_out, json_out])
    btn1.click(fn=clear_all, inputs=None, outputs=[img_in, img_out, json_out])
    gr.Button.style(1)

demo.launch(share=True)
