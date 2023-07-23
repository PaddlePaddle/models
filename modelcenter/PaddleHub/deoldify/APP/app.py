import paddlehub as hub


module = hub.Module(name='deoldify')
module.create_gradio_app().launch()
