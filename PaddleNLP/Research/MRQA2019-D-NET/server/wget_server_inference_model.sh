wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/D-Net/mrqa2019_inference_model.tar.gz
tar -xvf mrqa2019_inference_model.tar.gz
rm mrqa2019_inference_model.tar.gz
mv bert_infer_model bert_server/infer_model
mv xlnet_infer_model xlnet_server/infer_model
mv ernie_infer_model ernie_server/infer_model
