##                       类型作业演示                        ##
## 请将下面的 user_ak/user_sk 替换成自己的 ak/sk             ##
## 请将下面的 cluster_name 替换成所在组关联的k8s集群名称     ##
##                                                           ##
###############################################################
# 请替换成所在组下的个人 access key/secret key
#AK="55353f9e03565336b6b88a016d19ccc9"
#SK="0e3cb65d853d51eaa473855ce5c504f1"
AK="83543bd02884567f88a779f124e6fd20"
SK="f56ab9b874f65bf9a3340cf5e6d2a4ae"
# 作业参数
gpus_per_node="8"
k8s_gpu_type="baidu/gpu_v100"
k8s_wall_time="99:00:00"
k8s_memory="490Gi"
k8s_priority="high"
is_standalone="0"
k8s_trainers="4"
#batch_size=1024
batch_size=4096
JOB_NAME="ResNeXt101_vd_32x4d_n${k8s_trainers}_bs${batch_size}"
# 请替换成所在组关联的集群名称
cluster_name="v100-32-0-cluster"
group_name="dltp-0-yq01-k8s-gpu-v100-8"
# 作业版本
job_version="paddle-fluid-v1.6.0"

job_name=${JOB_NAME}
# 线上正式环境
server="paddlecloud.baidu-int.com"
port=80

distributed_conf="1 "
if [ ${k8s_trainers} -gt 1 ]
then
    distributed_conf="0 --distribute-job-type NCCL2 "
fi

upload_files="before_hook.sh end_hook.sh ../train.py"

# 启动命令
start_cmd="python -m paddle.distributed.launch \
              --use_paddlecloud \
              --selected_gpus="0,1,2,3,4,5,6,7" \
              --log_dir=mylog \
              train.py \
	            --model=ResNeXt101_vd_32x4d \
              --batch_size=$batch_size \
              --lr_strategy=cosine_decay \
              --warm_up_epochs=5.0 \
              --lr=1.6 \
              --num_epochs=200 \
              --model_save_dir=output/ \
              --l2_decay=1e-4 \
              --use_mixup=True \
              --use_label_smoothing=True \
              --label_smoothing_epsilon=0.1"

paddlecloud job train \
    --job-name ${job_name} \
    --group-name ${group_name} \
    --cluster-name ${cluster_name} \
    --job-conf job.cfg \
    --start-cmd "${start_cmd}" \
    --files ${upload_files} \
    --job-version ${job_version}  \
    --k8s-gpu-cards $gpus_per_node \
    --k8s-wall-time ${k8s_wall_time} \
    --k8s-memory ${k8s_memory} \
    --k8s-cpu-cores 35 \
    --k8s-trainers ${k8s_trainers} \
    --k8s-priority ${k8s_priority} \
    --is-standalone ${distributed_conf} 
