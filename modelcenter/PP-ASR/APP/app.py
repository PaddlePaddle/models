import gradio as gr
import os
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.text.infer import TextExecutor
import librosa
import soundfile as sf


def model_inference(audio):
    asr = ASRExecutor()
    text_punc = TextExecutor()
    if not isinstance(audio, str):
        audio = str(audio.name)
    y, sr = librosa.load(audio)
    if sr != 16000:  # Optional resample to 16000
        y = librosa.resample(y, sr, 16000)
        sf.write(audio, y, 16000)
    result = asr(audio_file=audio,
                 model='conformer_online_wenetspeech',
                 device="cpu")
    result = text_punc(
        text=result, model='ernie_linear_p7_wudao', device="cpu")
    return result


def clear_all():
    return None, None, None, None


with gr.Blocks() as demo:
    gr.Markdown("ASR")

    with gr.Column(scale=1, min_width=100):
        audio_input = gr.Audio(
            value='https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav',
            type="file",
            label=" Input From File")
        micro_input = gr.inputs.Audio(
            source="microphone", type='filepath', label="Input From Mic")

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit File")
            btn3 = gr.Button("Submit Micro")

        audio_text_output = gr.Textbox(placeholder="Result...", lines=10)
        micro_text_output = gr.Textbox(placeholder="Micro Result...", lines=10)

    btn3.click(
        fn=model_inference,
        inputs=[micro_input],
        outputs=micro_text_output,
        scroll_to_output=True)
    btn2.click(
        fn=model_inference,
        inputs=[audio_input],
        outputs=audio_text_output,
        scroll_to_output=True)
    btn1.click(
        fn=clear_all,
        inputs=None,
        outputs=[
            audio_input, micro_input, audio_text_output, micro_text_output
        ])

    gr.Button.style(1)

demo.launch()
