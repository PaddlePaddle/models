import paddlehub as hub


module = hub.Module(name='pp-tinypose')
module.create_gradio_app().launch()
