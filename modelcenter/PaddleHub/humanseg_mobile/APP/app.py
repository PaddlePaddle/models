import paddlehub as hub


module = hub.Module(name='humanseg_mobile')
module.create_gradio_app().launch()
