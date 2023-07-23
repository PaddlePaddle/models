import paddlehub as hub


module = hub.Module(name='humanseg_server')
module.create_gradio_app().launch()
