import gradio as gr
from paddlenlp import Taskflow

schema = ['时间', '选手', '赛事名称']
ie = Taskflow('information_extraction', schema=schema)


# UGC: Define the inference fn() for your models
def model_inference(schema, text):

    ie.set_schema(eval(schema))
    res = ie(text)
    json_out = {"text": text, "result": res}
    return json_out


def clear_all():
    return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("ERNIE-UIE")

    with gr.Column(scale=1, min_width=100):
        schema = gr.Textbox(
            placeholder="ex. ['时间', '选手', '赛事名称']",
            label="Type any schema you want:",
            lines=2)
        text = gr.Textbox(
            placeholder="ex. 2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！",
            label="Input Sequence:",
            lines=2)

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")
        json_out = gr.JSON(label="Information Extraction Output")

    btn1.click(fn=clear_all, inputs=None, outputs=[schema, text, json_out])
    btn2.click(fn=model_inference, inputs=[schema, text], outputs=[json_out])
    gr.Button.style(1)

demo.launch()
