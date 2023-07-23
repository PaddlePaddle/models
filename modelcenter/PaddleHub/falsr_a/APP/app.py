import paddlehub as hub


module = hub.Module(name='falsr_a')
module.create_gradio_app().launch()
