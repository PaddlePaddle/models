import paddlehub as hub


module = hub.Module(name='falsr_b')
module.create_gradio_app().launch()
