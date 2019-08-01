

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_memory_fraction_of_eager_deletion=1
export FLAGS_fast_eager_deletion_mode=1


export CUDA_VISIBLE_DEVICES=0 
python  train.py \
--train_path='data/train/sentence_file_*'  \
--test_path='data/dev/sentence_file_*'  \
--vocab_path data/vocabulary_min5k.txt \
--learning_rate 0.2 \
--use_gpu True \
--all_train_tokens 35479 \
--load_pretraining_params "checkpoints/4" \
--local True $@
