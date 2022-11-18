#-*- coding: UTF-8 -*-
# Copyright 2022 The Impira Team and the HuggingFace Team.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import base64
from io import BytesIO
from PIL import Image
import traceback

import requests
import numpy as np
import gradio as gr
import pdf2image
import fitz
import cv2

fitz_tools = fitz.Tools()


def pdf2img(stream, pagenos, dpi=300, thread_count=3, height=1600):
    images = []
    cimages = pdf2image.convert_from_bytes(
        stream, dpi=dpi, thread_count=thread_count, first_page=pagenos[0] + 1, last_page=pagenos[-1] + 1,
        size=height)
    for _image in cimages:
        image = np.array(_image)
        image = image[..., ::-1]
        images.append(image)
    return images


class PdfReader(object):
    """pdf reader"""
    def __init__(self,
                 stream: bytes,
                 image_height: int = 1600):
        self.stream = stream
        self._image_height = image_height
        self._dpi = 200
        self._inpdf = self.load_file(stream)

    @staticmethod
    def load_file(stream):
        """load document"""
        try:
            inpdf = fitz.Document(stream=stream, filetype="pdf")
        except Exception as e:
            print(f"[PDF_READER]-[Failed to load the file]-[{repr(e)}]")
        return inpdf

    @staticmethod
    def _convert_page_obj_to_image(page_obj, image_height: int = None):
        """fitz convert pdf to image

        Args:
            page_obj ([type]): [description]
            ratio ([type]): [description]

        Returns:
            [type]: [description]
        """
        if image_height:
            _, page_height = page_obj.rect.x1 - \
                page_obj.rect.x0, page_obj.rect.y1 - page_obj.rect.y0
            ratio = image_height / page_height
        else:
            ratio = 1.0
        trans = fitz.Matrix(ratio, ratio)
        pixmap = page_obj.get_pixmap(matrix=trans, alpha=False)
        image = cv2.imdecode(np.frombuffer(pixmap.tobytes(), np.uint8), -1)
        fitz_tools.store_shrink(100)
        return image

    def get_page_image(self,
                       pageno):
        """get page image

        Args:
            pageno ([type]): [description]

        Returns:
            [type]: [description]
        """
        try:
            page_obj = self._inpdf[pageno]
            return self._convert_page_obj_to_image(page_obj, self._image_height)
        except Exception as e:
            print(f"[Failed to convert the PDF to images]-[{repr(e)}]")
        try:
            return pdf2img(stream=self.stream,
                           pagenos=[pageno],
                           height=self._image_height,
                           dpi=self._dpi)[0]
        except Exception as e:
            print(f"[Failed to convert the PDF to images]-[{repr(e)}]")
        return None


examples = [
    [
        "budget_form.png",
        "What is the total actual and/or obligated expenses of ECG Center?"
    ],
    [
        "poster.png",
        "Which gift idea needs a printer?"
    ],
        [
        "receipt.png",
        "เบอร์โทรร้านอะไรคะ?"
    ],
    [
        "medical_bill_2.jpg",
        "患者さんは何でお金を払いますか。"
    ],
    [
        "resume.png",
        "五百丁本次想要担任的是什么职位?",
    ],
    [
        "custom_declaration_form.png",
        "在哪个口岸进口？"
    ],
    [
        "invoice.jpg",
        "发票号码是多少?",
    ],
]

prompt_files = {
    "发票号码是多少?": "invoice.jpg",
    "五百丁本次想要担任的是什么职位?": "resume.png",
    "在哪个口岸进口？": "custom_declaration_form.png",
    "What is the total actual and/or obligated expenses of ECG Center?": "budget_form.png",
    "Which gift idea needs a printer?": "poster.png",
    "患者さんは何でお金を払いますか。": "medical_bill_2.jpg",
    "เบอร์โทรร้านอะไรคะ?": "receipt.png",
}

lang_map = {
    "invoice.jpg": "ch",
    "resume.png": "ch",
    "custom_declaration_form.png": "ch",
    "medical_bill_1.png": "ch",
    "budget_form.png": "en",
    "website_design_guide.jpeg": "en",
    "poster.png": "en",
    "medical_bill_2.jpg": "ch",
    "receipt.png": "en"
}


def load_document(path):
    if path.startswith("http://") or path.startswith("https://"):
        resp = requests.get(path, allow_redirects=True, stream=True)
        b = resp.raw
    else:
        b = open(path, "rb")

    if path.endswith(".pdf"):
        images_list = []
        pdfreader = PdfReader(stream=b.read())
        for p_no in range(0, pdfreader._inpdf.page_count):
            img_np = pdfreader.get_page_image(pageno=p_no)
            images_list.append(img_np)
    else:
        image = Image.open(b)
        images_list = [np.array(image.convert("RGB"))]
    return images_list

def process_path(path):
    error = None
    if path:
        try:
            images_list = load_document(path)
            return (
                path,
                gr.update(visible=True, value=images_list),
                gr.update(visible=True),
                gr.update(visible=False, value=None),
                gr.update(visible=False, value=None),
                None,
            )
        except Exception as e:
            traceback.print_exc()
            error = str(e)
    return (
        None,
        gr.update(visible=False, value=None),
        gr.update(visible=False),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=True, value=error) if error is not None else None,
        None,
    )


def process_upload(file):
    if file:
        return process_path(file.name)
    else:
        return (
            None,
            gr.update(visible=False, value=None),
            gr.update(visible=False),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            None,
        )


def np2base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    base64_str = str(base64.b64encode(image))[2:-1]
    return base64_str


def get_base64(path):
    if path.startswith("http://") or path.startswith("https://"):
        resp = requests.get(path, allow_redirects=True, stream=True)
        b = resp.raw
    else:
        b = open(path, "rb")

    if path.endswith(".pdf"):
        images_list = []
        pdfreader = PdfReader(stream=b.read())
        for p_no in range(0, min(pdfreader._inpdf.page_count, 1)):
            img_np = pdfreader.get_page_image(pageno=p_no)
            images_list.append(img_np)
        base64_str = np2base64(images_list[0])
    else:
        base64_str = base64.b64encode(b.read()).decode()
    return base64_str


def process_prompt(prompt, document, lang="ch"):
    if not prompt:
        prompt = "What is the total actual and/or obligated expenses of ECG Center?"
    if document is None:
        return None, None, None

    access_token = os.environ['token']

    url = f"https://aip.baidubce.com/rpc/2.0/nlp-itec/poc/docprompt?access_token={access_token}"
    
    base64_str = get_base64(document)

    r = requests.post(url, json={"doc": base64_str, "prompt": [prompt], "lang": lang})
    response = r.json()
    
    predictions = response['result']
    img_list = response['image']
    pages = [Image.open(BytesIO(base64.b64decode(img))) for img in img_list]

    text_value = predictions[0]['result'][0]['value']

    return (
        gr.update(visible=True, value=pages),
        gr.update(visible=True, value=predictions),
        gr.update(
            visible=True,
            value=text_value,
        ),
    )


def load_example_document(img, prompt):
    if img is not None:
        document = prompt_files[prompt]
        lang = lang_map[document]
        preview, answer, answer_text = process_prompt(prompt, document, lang)
        return document, prompt, preview, gr.update(visible=True), answer, answer_text
    else:
        return None, None, None, gr.update(visible=False), None, None


def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


CSS = """
#prompt input {
    font-size: 16px;
}
#url-textbox {
    padding: 0 !important;
}
#short-upload-box .w-full {
    min-height: 10rem !important;
}
/* I think something like this can be used to re-shape
 * the table
 */
/*
.gr-samples-table tr {
    display: inline;
}
.gr-samples-table .p-2 {
    width: 100px;
}
*/
#select-a-file {
    width: 100%;
}
#file-clear {
    padding-top: 2px !important;
    padding-bottom: 2px !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
	margin-top: 10px;
}
.gradio-container .gr-button-primary {
    background: linear-gradient(180deg, #CDF9BE 0%, #AFF497 100%);
    border: 1px solid #B0DCCC;
    border-radius: 8px;
    color: #1B8700;
}
.gradio-container.dark button#submit-button {
    background: linear-gradient(180deg, #CDF9BE 0%, #AFF497 100%);
    border: 1px solid #B0DCCC;
    border-radius: 8px;
    color: #1B8700
}
table.gr-samples-table tr td {
    border: none;
    outline: none;
}
table.gr-samples-table tr td:first-of-type {
    width: 0%;
}
div#short-upload-box div.absolute {
    display: none !important;
}
gradio-app > div > div > div > div.w-full > div, .gradio-app > div > div > div > div.w-full > div {
    gap: 0px 2%;
}
gradio-app div div div div.w-full, .gradio-app div div div div.w-full {
    gap: 0px;
}
gradio-app h2, .gradio-app h2 {
    padding-top: 10px;
}
#answer {
    overflow-y: scroll;
    color: white;
    background: #666;
    border-color: #666;
    font-size: 20px;
    font-weight: bold;
}
#answer span {
    color: white;
}
#answer textarea {
    color:white;
    background: #777;
    border-color: #777;
    font-size: 18px;
}
#url-error input {
    color: red;
}
"""

with gr.Blocks(css=CSS) as demo:
    gr.HTML(read_content("header.html"))
    gr.Markdown(
        "DocPrompt🔖 is a Document Prompt Engine using ERNIE-Layout as the backbone model."
        "The engine is powered by BAIDU WenXin Document Intelligence Team "
        "and has the ability for multilingual documents information extraction and question ansering. "
        "For more details, please visit the [Github](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-layout)."
        "ERNIE-Layout paper please refer to [ERNIE-Layout](https://paperswithcode.com/paper/ernie-layout-layout-knowledge-enhanced-pre)"
    )

    document = gr.Variable()
    example_prompt = gr.Textbox(visible=False)
    example_image = gr.Image(visible=False)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                gr.Markdown("## 1. Select a file", elem_id="select-a-file")
                img_clear_button = gr.Button(
                    "Clear", variant="secondary", elem_id="file-clear", visible=False
                )
            image = gr.Gallery(visible=False)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        url = gr.Textbox(
                            show_label=False,
                            placeholder="URL",
                            lines=1,
                            max_lines=1,
                            elem_id="url-textbox",
                        )
                        submit = gr.Button("Get")
                    url_error = gr.Textbox(
                        visible=False,
                        elem_id="url-error",
                        max_lines=1,
                        interactive=False,
                        label="Error",
                    )
            gr.Markdown("— or —")
            upload = gr.File(label=None, interactive=True, elem_id="short-upload-box")
            gr.Examples(
                examples=examples,
                inputs=[example_image, example_prompt],
            )

        with gr.Column() as col:
            gr.Markdown("## 2. Make a request")
            prompt = gr.Textbox(
                label="Prompt (No restrictions on the setting of prompt. You can type any prompt.)",
                placeholder="e.g. What is the total actual and/or obligated expenses of ECG Center?",
                lines=1,
                max_lines=1,
            )
            ocr_lang = gr.Radio(
                choices=["ch", "en"],
                value="en",
                label="Select OCR Language (Please choose ch for Chinese images.)",
            )

            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary")
                submit_button = gr.Button(
                    "Submit", variant="primary", elem_id="submit-button"
                )
            with gr.Column():
                output_text = gr.Textbox(
                    label="Top Answer", visible=False, elem_id="answer"
                )
                output = gr.JSON(label="Output", visible=False)

    for cb in [img_clear_button, clear_button]:
        cb.click(
            lambda _: (
                gr.update(visible=False, value=None),
                None,
                gr.update(visible=False, value=None),
                gr.update(visible=False, value=None),
                gr.update(visible=False),
                None,
                None,
                None,
                gr.update(visible=False, value=None),
                None,
            ),
            inputs=clear_button,
            outputs=[
                image,
                document,
                output,
                output_text,
                img_clear_button,
                example_image,
                upload,
                url,
                url_error,
                prompt,
            ],
        )

    upload.change(
        fn=process_upload,
        inputs=[upload],
        outputs=[document, image, img_clear_button, output, output_text, url_error],
    )
    submit.click(
        fn=process_path,
        inputs=[url],
        outputs=[document, image, img_clear_button, output, output_text, url_error],
    )

    prompt.submit(
        fn=process_prompt,
        inputs=[prompt, document, ocr_lang],
        outputs=[image, output, output_text],
    )

    submit_button.click(
        fn=process_prompt,
        inputs=[prompt, document, ocr_lang],
        outputs=[image, output, output_text],
    )

    example_image.change(
        fn=load_example_document,
        inputs=[example_image, example_prompt],
        outputs=[document, prompt, image, img_clear_button, output, output_text],
    )

    gr.Markdown("[![Stargazers repo roster for @PaddlePaddle/PaddleNLP](https://reporoster.com/stars/PaddlePaddle/PaddleNLP)](https://github.com/PaddlePaddle/PaddleNLP)")
    gr.HTML(read_content("footer.html"))


if __name__ == "__main__":
    demo.launch(enable_queue=False,share=True)