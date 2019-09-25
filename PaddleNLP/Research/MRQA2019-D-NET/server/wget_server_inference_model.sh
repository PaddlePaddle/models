wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/D-Net/mrqa2019_inference_model.tar.gz
tar -xvf mrqa2019_inference_model.tar.gz
rm mrqa2019_inference_model.tar.gz
mv infer_model bert_server
mv infer_model_800_bs128 xlnet_server
