import paddlehub as hub


module = hub.Module(name='chinese_ocr_db_crnn_server')
module.create_gradio_app().launch()
