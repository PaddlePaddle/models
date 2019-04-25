export FLAGS_eager_delete_tensor_gb=0.0

#pretrain
python -u main.py \
  --do_train True \
  --sample_pro 0.9 \
  --batch_size 64 \
  --save_path model_files_tmp/matching_pretrained \
  --train_path data/unlabel_data/train.ids \
  --val_path data/unlabel_data/val.ids

#finetune based on one task
TASK=human
python -u main.py \
  --do_train True \
  --loss_type L2 \
  --save_path model_files_tmp/${TASK}_finetuned \
  --init_model model_files/matching_pretrained \
  --train_path data/label_data/$TASK/train.ids \
  --val_path data/label_data/$TASK/val.ids \
  --print_step 1 \
  --save_step 1 \
  --num_scan_data 50 

#evaluate pretrained model by Recall
python -u main.py \
  --do_val True \
  --test_path data/unlabel_data/test.ids \
  --init_model model_files/matching_pretrained \
  --loss_type CLS

#evaluate pretrained model by Cor
for task in seq2seq_naive seq2seq_att keywords human
do
  echo $task
  python -u main.py \
    --do_val True \
    --test_path data/label_data/$task/test.ids \
    --init_model model_files/matching_pretrained \
    --loss_type L2
done

#evaluate finetuned model by Cor
for task in seq2seq_naive seq2seq_att keywords human
do
  echo $task
  python -u main.py \
    --do_val True \
    --test_path data/label_data/$task/test.ids \
    --init_model model_files/${task}_finetuned \
    --loss_type L2 
done
  
#infer
TASK=human
python -u main.py \
  --do_infer True \
  --test_path data/label_data/$TASK/test.ids \
  --init_model model_files/${TASK}_finetuned
