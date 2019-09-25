# wget train data
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/D-Net/mrqa_multi_task_dataset.tar.gz
tar -xvf mrqa_multi_task_dataset.tar.gz
rm mrqa_multi_task_dataset.tar.gz

# wget predictions results
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/D-Net/muiti_task_prediction_results.tar.gz
tar -xvf muiti_task_prediction_results.tar.gz
rm muiti_task_prediction_results.tar.gz
