import paddlehub as hub


module = hub.Module(name='transformer_zh-en')
module.create_gradio_app().launch()
