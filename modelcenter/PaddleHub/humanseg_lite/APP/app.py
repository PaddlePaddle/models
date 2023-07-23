import paddlehub as hub


module = hub.Module(name='humanseg_lite')
module.create_gradio_app().launch()
