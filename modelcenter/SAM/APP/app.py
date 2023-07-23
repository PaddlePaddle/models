import gradio as gr
import numpy as np
import cv2

import utils
from predict import build_predictor

ID_PHOTO_IMAGE_DEMO = "./images/cityscapes_demo.png"

generator = build_predictor()


def clear_image_all():
    utils.delete_result()
    return None, None, None, None


def get_id_photo_output(img):
    """
    Get the special size and background photo.

    Args:
        img(numpy:ndarray): The image array.
        size(str): The size user specified.
        bg(str): The background color user specified.
        download_size(str): The size for image saving.

    """
    predictor = generator
    masks = predictor.generate(img)
    pred_result, pseudo_map = utils.masks2pseudomap(masks)  # PIL Image
    added_pseudo_map = utils.visualize(
        img, pred_result, color_map=utils.get_color_map_list(256))
    res_download = utils.download(pseudo_map)

    return pseudo_map, added_pseudo_map, res_download


with gr.Blocks() as demo:
    gr.Markdown("""# Segment Anything (PaddleSeg) """)
    with gr.Tab("InputImage"):
        image_in = gr.Image(value=ID_PHOTO_IMAGE_DEMO, label="Input image")

        with gr.Row():
            image_clear_btn = gr.Button("Clear")
            image_submit_btn = gr.Button("Submit")

        with gr.Row():
            img_out1 = gr.Image(
                label="Output image", interactive=False).style(height=300)
            img_out2 = gr.Image(
                label="Output image with mask",
                interactive=False).style(height=300)
        downloaded_img = gr.File(label='Image download').style(height=50)

    image_clear_btn.click(
        fn=clear_image_all,
        inputs=None,
        outputs=[image_in, img_out1, img_out2, downloaded_img])

    image_submit_btn.click(
        fn=get_id_photo_output,
        inputs=[image_in, ],
        outputs=[img_out1, img_out2, downloaded_img])

    gr.Markdown(
        """<font color=Gray>Tips: You can try segment the default image OR upload any images you want to segment by click on the clear button first.</font>"""
    )

    gr.Markdown(
        """<font color=Gray>This is Segment Anything build with PaddlePaddle. 
        We refer to the [SAM](https://github.com/facebookresearch/segment-anything) for code strucure and model architecture.
        If you have any question or feature request, welcome to raise issues on [GitHub](https://github.com/PaddlePaddle/PaddleSeg/issues). </font>"""
    )

    gr.Button.style(1)

demo.launch(server_name="0.0.0.0", server_port=8021, share=True)
