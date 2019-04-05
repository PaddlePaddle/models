
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=3
export FLAGS_fraction_of_gpu_memory_to_use=0.95
TASK_NAME='senta'
DATA_PATH=./senta_data/train_set/
CKPT_PATH=./save_models

# run_train
fluid -u run_classifier.py --task_name ${TASK_NAME} --use_cuda true --do_train true --do_val true --do_infer true --batch_size 256 --data_dir ${DATA_PATH} --vocab_path senta_data/word_dict.txt --checkpoints ${CKPT_PATH} --save_steps 1000 --validation_steps 25 --epoch 3 --senta_config_path ./senta_config.json --skip_steps 10

DATA_PATH=./scdb_data/test_set/
# run_eval
fluid -u run_classifier.py --task_name senta --use_cuda true --do_train false --do_val true --do_infer true --batch_size 10 --data_dir ${DATA_PATH} --vocab_path scdb_data/train_set/network.vob.u8 --init_checkpoint save_models/step_41/ --senta_config_path ./senta_config.json
