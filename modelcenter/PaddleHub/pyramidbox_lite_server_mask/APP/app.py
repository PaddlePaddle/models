import paddlehub as hub


module = hub.Module(name='pyramidbox_lite_server_mask')
module.create_gradio_app().launch()
