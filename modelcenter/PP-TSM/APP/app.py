import gradio as gr

from ppvideo import PaddleVideo


TOPK = 5
pv = PaddleVideo(model_name='ppTSM', use_gpu=False, top_k=TOPK)


# Define the model inference
def model_inference(video_file):
    result = pv.predict(video_file)[0][0]
    topk_scores = result['topk_scores']
    label_names = result['label_names']
    
    output = {label_names[i]: float(topk_scores[i]) for i in range(TOPK)}
    
    return output


def clear_all():
    return None, None


with gr.Blocks() as demo:
    gr.Markdown("PP-TSM")

    with gr.Column(scale=1, min_width=100):
        video_in = gr.Video(
            value="https://videotag.bj.bcebos.com/Data/swim.mp4",
            label="Input (Some formats cannot be previewed)")

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")
        outputs = gr.outputs.Label(num_top_classes=TOPK)

    btn2.click(fn=model_inference, inputs=video_in, outputs=outputs)
    btn1.click(fn=clear_all, inputs=None, outputs=[video_in, outputs])
    gr.Button.style(1)

demo.launch()