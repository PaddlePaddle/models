import paddlehub as hub


module = hub.Module(name='pyramidbox_lite_server')
module.create_gradio_app().launch()
