import paddlehub as hub


module = hub.Module(name='falsr_c')
module.create_gradio_app().launch()
