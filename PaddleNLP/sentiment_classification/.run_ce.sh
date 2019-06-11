#! /bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export CPU_NUM=1


# run_train on train.tsv and do_val on test.tsv
train() {
    python -u run_classifier.py \
        --task_name 'senta' \
        --use_cuda true \
        --do_train true \
        --do_val true \
        --do_infer false \
        --batch_size 16 \
        --data_dir ./senta_data/ \
        --vocab_path ./senta_data/word_dict.txt \
        --checkpoints ./save_models \
        --save_steps 500 \
        --validation_steps 50 \
        --epoch 2 \
        --senta_config_path ./senta_config.json \
        --skip_steps 10 \
        --enable_ce
}

train_ernie() {
    python -u run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_train true \
        --do_val true \
        --do_infer false \
        --batch_size 24 \
        --init_checkpoint ./senta_model/ernie_pretrain_model/params \
        --train_set ./senta_data/train.tsv \
        --dev_set ./senta_data/dev.tsv \
        --test_set ./senta_data/test.tsv \
        --vocab_path ./senta_model/ernie_pretrain_model/vocab.txt \
        --checkpoints ./save_models \
        --save_steps 5000 \
        --validation_steps 100 \
        --epoch 2 \
        --max_seq_len 256 \
        --ernie_config_path ./senta_model/ernie_pretrain_model/ernie_config.json \
        --model_type "ernie_base" \
        --lr 5e-5 \
        --skip_steps 10 \
        --num_labels 2 \
        --random_seed 1
}

export CUDA_VISIBLE_DEVICES=0
train | grep "dev evaluation" | grep "ave loss" | tail -1 | awk '{print "kpis\ttrain_loss_senta_card1\t"$5"\nkpis\ttrain_acc_senta_card1\t"$8"\nkpis\teach_step_duration_senta_card1\t"$11}' | tr -d "," | python _ce.py
sleep 20

export CUDA_VISIBLE_DEVICES=0,1,2,3
train | grep "dev evaluation" | grep "ave loss" | tail -1 | awk '{print "kpis\ttrain_loss_senta_card4\t"$5"\nkpis\ttrain_acc_senta_card4\t"$8"\nkpis\teach_step_duration_senta_card4\t"$11}' | tr -d "," | python _ce.py
