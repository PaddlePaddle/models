import os

import gradio as gr
from paddlespeech.cli.tts import TTSExecutor

tts_executor = TTSExecutor()


def speech_generate(text: str) -> os.PathLike:
    assert isinstance(text,
                      str) and len(text) > 0, 'Input Chinese-English text...'
    wav_file = tts_executor(
        text=text,
        output='output.wav',
        am='fastspeech2_mix',
        voc='hifigan_csmsc',
        lang='mix',
        spk_id=174)
    return wav_file


def clear_all():
    return None, None


with gr.Blocks() as demo:
    gr.Markdown("Text to Speech")

    with gr.Column(scale=1, min_width=50):
        text_input = gr.Textbox(placeholder="Type here...", lines=5)

        with gr.Row():
            btn1 = gr.Button("Clear")
            btn2 = gr.Button("Submit")

        audio_output = gr.Audio(type="file", label="Output")

    btn2.click(
        fn=speech_generate,
        inputs=text_input,
        outputs=audio_output,
        scroll_to_output=True)
    btn1.click(fn=clear_all, inputs=None, outputs=[text_input, audio_output])

    gr.Button.style(1)

demo.launch()
