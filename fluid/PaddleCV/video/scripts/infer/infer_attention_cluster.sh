python infer.py --model-name="AttentionCluster" --config=./configs/attention_cluster.txt \
                --filelist=./data/youtube8m/infer.list \
                --weights=./checkpoints/AttentionCluster_epoch0 \
                --save-dir="./save"
