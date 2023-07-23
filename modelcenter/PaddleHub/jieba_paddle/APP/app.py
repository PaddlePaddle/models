import paddlehub as hub


module = hub.Module(name='jieba_paddle')
module.create_gradio_app().launch()
