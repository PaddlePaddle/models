import gradio as gr
import numpy as np

import utils
from predict import build_predictor

IMAGE_DEMO = "./images/idphoto.jpg"
predictor = build_predictor()
sizes_play = utils.size_play()


def get_output(img, size, bg, download_size):
    """
    Get the special size and background photo.

    Args:
        img(numpy:ndarray): The image array.
        size(str): The size user specified.
        bg(str): The background color user specified.
        download_size(str): The size for image saving.

    """
    alpha = predictor.run(img)
    res = utils.bg_replace(img, alpha, bg_name=bg)

    size_index = sizes_play.index(size)
    res = utils.adjust_size(res, size_index)
    res_download = utils.download(res, download_size)
    return res, res_download


def download(img, size):
    utils.download(img, size)
    return None


with gr.Blocks() as demo:
    gr.Markdown("""# ID Photo DIY""")

    img_in = gr.Image(value=IMAGE_DEMO, label="Input image")
    gr.Markdown(
        """<font color=Gray>Tips: Please upload photos with good posture, center portrait, crown free, no jewelry, ears and eyebrows exposed.</font>"""
    )
    with gr.Row():
        size = gr.Dropdown(sizes_play, label="Sizes", value=sizes_play[0])
        bg = gr.Radio(
            ["White", "Red", "Blue"], label="Background color", value='White')
        download_size = gr.Radio(
            ["Small", "Middle", "Large"],
            label="File size (affects image quality)",
            value='Large',
            interactive=True)

    with gr.Row():
        btn1 = gr.Button("Clear")
        btn2 = gr.Button("Submit")

    img_out = gr.Image(
        label="Output image", interactive=False).style(height=300)
    f1 = gr.File(label='Image download').style(height=50)
    with gr.Row():
        gr.Markdown(
            """<font color=Gray>This application is supported by [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg). 
            If you have any questions or feature requists, welcome to raise issues on [GitHub](https://github.com/PaddlePaddle/PaddleSeg/issues). BTW, a star is a great encouragement for us, thanks!  ^_^</font>"""
        )

    btn2.click(
        fn=get_output,
        inputs=[img_in, size, bg, download_size],
        outputs=[img_out, f1])
    btn1.click(
        fn=utils.clear_all,
        inputs=None,
        outputs=[img_in, img_out, size, bg, download_size, f1])

    gr.Button.style(1)

demo.launch(share=True)
