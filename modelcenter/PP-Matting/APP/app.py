import gradio as gr
import numpy as np
import cv2

import utils
from predict import build_predictor

ID_PHOTO_IMAGE_DEMO = "./images/idphoto.jpg"

MC_PHOTO_BG = "Red"
MC_PHOTO_SIZE = (413, 626)
MC_PHOTO_WIFE_IMAGE_DEMO = "./images/wife.jpg"
MC_PHOTO_HUSBAND_IMAGE_DEMO = "./images/husband.jpg"

predictor = build_predictor()
sizes_play = utils.size_play()


def crop(img, alpha, thr=0.001):
    """
    Crop the image and alpha according to alpha mask.

    Args:
        img(numpy:ndarray): The image array.
        alpha(numpy:ndarray): The alpha corresponding to the image.
        thr(float): The threshold used to generate the alpha mask.

    Returns:
        img_cropped(numpy:ndarray): Image after crop.
        alpha_cropped(numpy:ndarray): alpha after crop.
    """
    _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY, thr)
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = cv2.boundingRect(cnt[0])
    return img[y: y + h, x: x + w], alpha[y: y + h, x: x + w]


def copy_img_to_bg(img, alpha, target_h, target_x, bg_color, bg_size):
    """
    Copy img and alpha to the corresponding background respectively.

    Args:
        img(numpy:ndarray): The image array.
        alpha(numpy:ndarray): The alpha corresponding to the image.
        target_h(int): The target height of img in background.
        target_x(int): The target position of the img's x-center in the background.
        bg_color(str): The color of background, the options are "White"、"Blue"、"Red".
        bg_size(tuple): The size of background.

    Returns:
        res_img(numpy:ndarray): Result after copy img to background.
        res_alpha(numpy:ndarray): Result after copy alpha to background.
    """
    bg_h, bg_w = bg_size
    img_h, img_w, _ = img.shape
    r = 1.0 * img_h / target_h
    target_w = int(img_w / r)

    res_img = np.ones((bg_h, bg_w, 3)) * utils.COLOR_MAP[bg_color]
    res_alpha = np.zeros((bg_h, bg_w))

    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)

    x1 = int(max(0, target_x - target_w / 2))
    y1 = int(max(0, bg_h - target_h))
    x2 = int(min(bg_w, x1+target_w))
    y2 = int(min(bg_h, y1+target_h))
    act_w = x2 - x1
    act_h = y2 - y1
    res_img[y1: y2, x1: x2] = img[:act_h, :act_w]
    res_alpha[y1: y2, x1: x2] = alpha[:act_h, :act_w]

    return res_img, res_alpha


def get_mc_photo_app_output(wife, husband):
    """
    Composite marriage certificate photo

    Args:
        wife(numpy:ndarray): The wife image array.
        husband(numpy:ndarray): The husband image array.

    Returns:
        res(numpy:ndarray): The composited marriage certificate photo.
        res_download(str): The path of res.
    """
    alpha_wife = predictor.run(wife)
    alpha_husband = predictor.run(husband)

    _, wife_fg = utils.bg_replace(wife, alpha_wife, bg_name=MC_PHOTO_BG)
    husband, _ = utils.bg_replace(husband, alpha_husband, bg_name=MC_PHOTO_BG)

    wife_fg, alpha_wife = crop(wife_fg, alpha_wife)
    husband, alpha_husband = crop(husband, alpha_husband)

    wife_fg, alpha_wife = copy_img_to_bg(wife_fg, alpha_wife, 338, 225, MC_PHOTO_BG, MC_PHOTO_SIZE)
    husband, alpha_husband = copy_img_to_bg(husband, alpha_husband, 375, 401, MC_PHOTO_BG, MC_PHOTO_SIZE)

    alpha_wife = alpha_wife[:, :, None] / 255.0
    res = (wife_fg * alpha_wife + (1 - alpha_wife) * husband).astype(np.uint8)

    res_download = utils.download(res, "Large")
    return res, res_download


def get_id_photo_output(img, size, bg, download_size):
    """
    Get the special size and background photo.

    Args:
        img(numpy:ndarray): The image array.
        size(str): The size user specified.
        bg(str): The background color user specified.
        download_size(str): The size for image saving.

    """
    alpha = predictor.run(img)
    res, _ = utils.bg_replace(img, alpha, bg_name=bg)

    size_index = sizes_play.index(size)
    res = utils.adjust_size(res, size_index)
    res_download = utils.download(res, download_size)
    return res, res_download


def clear_id_photo_all():
    utils.delete_result()
    return None, None, sizes_play[0], 'White', 'Large', None


def clear_mc_photo_all():
    utils.delete_result()
    return None, None, None, None


with gr.Blocks() as demo:
    gr.Markdown("""# ID and MC(Marriage Certificate) Photo DIY""")
    gr.Markdown(
        """<font color=Gray>Tips: Please upload photos with good posture, center portrait, 
        crown free, no jewelry, ears and eyebrows exposed.</font>"""
    )
    with gr.Tab("IDPhoto"):
        id_photo_img_in = gr.Image(value=ID_PHOTO_IMAGE_DEMO, label="Input image")
        with gr.Row():
            id_photo_size = gr.Dropdown(sizes_play, label="Sizes", value=sizes_play[0])
            id_photo_bg = gr.Radio(
                ["White", "Red", "Blue"], label="Background color", value='White')
            id_photo_download_size = gr.Radio(
                ["Small", "Middle", "Large"],
                label="File size (affects image quality)",
                value='Large',
                interactive=True)

        with gr.Row():
            id_photo_clear_btn = gr.Button("Clear")
            id_photo_submit_btn = gr.Button("Submit")

        id_photo_img_out = gr.Image(label="Output image", interactive=False).style(height=300)
        id_photo_downloaded_img = gr.File(label='Image download').style(height=50)

    id_photo_clear_btn.click(
        fn=clear_id_photo_all,
        inputs=None,
        outputs=[id_photo_img_in, id_photo_img_out, id_photo_size, id_photo_bg,
                 id_photo_download_size, id_photo_downloaded_img])
    id_photo_submit_btn.click(
        fn=get_id_photo_output,
        inputs=[id_photo_img_in, id_photo_size, id_photo_bg, id_photo_download_size],
        outputs=[id_photo_img_out, id_photo_downloaded_img])

    with gr.Tab("MCPhoto"):
        with gr.Row():
            mc_photo_img_wife = gr.Image(value=MC_PHOTO_WIFE_IMAGE_DEMO, label="Wife", interactive=True)
            mc_photo_img_husband = gr.Image(value=MC_PHOTO_HUSBAND_IMAGE_DEMO, label="Husband", interactive=True)

        with gr.Row():
            mc_photo_clear_button = gr.Button("Clear")
            mc_photo_submit_button = gr.Button("Submit")

        mc_photo_img_out = gr.Image(label="Output image", interactive=False).style(height=300)
        mc_photo_download_img = gr.File(label='Image download').style(height=50)

    mc_photo_clear_button.click(
        fn=clear_mc_photo_all,
        inputs=None,
        outputs=[mc_photo_img_wife, mc_photo_img_husband, mc_photo_img_out, mc_photo_download_img])
    mc_photo_submit_button.click(
        fn=get_mc_photo_app_output,
        inputs=[mc_photo_img_wife, mc_photo_img_husband],
        outputs=[mc_photo_img_out, mc_photo_download_img])

    gr.Markdown(
        """<font color=Gray>This application is supported by 
        [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg). 
        If you have any question or feature request, 
        welcome to raise issues on [GitHub](https://github.com/PaddlePaddle/PaddleSeg/issues). 
        BTW, a star is a great encouragement for us, thanks!  ^_^</font>"""
    )

    gr.Button.style(1)

demo.launch(share=True)
