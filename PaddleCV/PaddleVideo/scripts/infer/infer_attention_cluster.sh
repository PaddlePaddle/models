python infer.py --model_name="AttentionCluster" --config=./configs/attention_cluster.txt \
                --filelist=./dataset/youtube8m/infer.list \
                --weights=./checkpoints/AttentionCluster_epoch0 \
                --save_dir="./save"
