import paddlehub as hub


module = hub.Module(name='ernie_vilg')
module.create_gradio_app().launch()
