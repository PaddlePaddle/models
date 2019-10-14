# wget pretrain model
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/D-Net/squad2_model.tar.gz
tar -xvf squad2_model.tar.gz
rm squad2_model.tar.gz
mv squad2_model ./data/pretrain_model/

# wget knowledge_distillation dataset
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/D-Net/d_net_knowledge_distillation_dataset.tar.gz
tar -xvf d_net_knowledge_distillation_dataset.tar.gz
rm d_net_knowledge_distillation_dataset.tar.gz
mv mlm_data ./data/input
mv mrqa_distill_data ./data/input

# wget evaluation dev dataset
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/D-Net/mrqa_evaluation_dataset.tar.gz 
tar -xvf mrqa_evaluation_dataset.tar.gz 
rm mrqa_evaluation_dataset.tar.gz 
mv mrqa_evaluation_dataset ./data/input

# wget predictions results
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/D-Net/kd_prediction_results.tar.gz
tar -xvf kd_prediction_results.tar.gz
rm kd_prediction_results.tar.gz

# wget MRQA baidu trained knowledge distillation model
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/D-Net/knowledge_distillation_model.tar.gz
tar -xvf knowledge_distillation_model.tar.gz
rm knowledge_distillation_model.tar.gz
mv knowledge_distillation_model ./data/saved_models




