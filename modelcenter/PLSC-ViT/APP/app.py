import gradio as gr
from predictor import Predictor

model_path = "paddlecv://models/vit/v2.4/imagenet2012-ViT-B_16-224_infer.pdmodel"
params_path = "paddlecv://models/vit/v2.4/imagenet2012-ViT-B_16-224_infer.pdiparams"
label_path = "paddlecv://dataset/imagenet2012_labels.txt"

predictor = None


def model_inference(image):
    global predictor
    if predictor is None:
        predictor = Predictor(
            model_path=model_path,
            params_path=params_path,
            label_path=label_path)
    scores, labels = predictor.predict(image)
    json_out = {"scores": scores.tolist(), "labels": labels.tolist()}
    return image, json_out


def clear_all():
    return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("Classification based on ViT")

    with gr.Column(scale=1, min_width=100):

        img_in = gr.Image(
            value="https://plsc.bj.bcebos.com/dataset/test_images/cat.jpg",
            label="Input")

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")

        img_out = gr.Image(label="Output")
        json_out = gr.JSON(label="jsonOutput")

    btn2.click(fn=model_inference, inputs=img_in, outputs=[img_out, json_out])
    btn1.click(fn=clear_all, inputs=None, outputs=[img_in, img_out, json_out])
    gr.Button.style(1)

demo.launch()
