import paddlehub as hub


module = hub.Module(name='solov2')
module.create_gradio_app().launch()
