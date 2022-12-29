import paddlehub as hub


module = hub.Module(name='ch_pp-ocrv3')
module.create_gradio_app().launch()
