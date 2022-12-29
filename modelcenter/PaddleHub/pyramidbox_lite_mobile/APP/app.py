import paddlehub as hub


module = hub.Module(name='pyramidbox_lite_mobile')
module.create_gradio_app().launch()
