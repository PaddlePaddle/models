import paddlehub as hub


module = hub.Module(name='lac')
module.create_gradio_app().launch()
