DIRS=`pwd`
data_path="${DIRS}/data"
train_files_path="${data_path}/train"
test_files_path="${data_path}/test"

model_files_path="${DIRS}/model"
# init_model will download in thirdparty folder when do paddlecloud training
init_model_files_path="${model_files_path}/init_model"

thirdparty_path="${DIRS}/thirdparty"
tree_travel_init_path="${thirdparty_path}/travel_list.txt"
tree_layer_init_path="${thirdparty_path}/layer_list.txt"
tree_info_init_path="${thirdparty_path}/tree_info.txt"

python_bin="python"
echo `pwd`

function main() {
    cmd="${python_bin} distributed_train.py \
        --is_cloud=0 \
        --sync_mode=async \
        --cpu_num=1 \
        --random_seed=0 \
        --epoch_num=1 \
        --batch_size=32 \
        --learning_rate=3e-4 \
        --train_files_path=${train_files_path} \
        --test_files_path=${test_files_path} \
        --model_files_path=${model_files_path} \
        --init_model_files_path=${init_model_files_path} \
        --tree_travel_init_path=${tree_travel_init_path} \
        --tree_info_init_path=${tree_info_init_path} \
        --tree_layer_init_path=${tree_layer_init_path} "
    echo ${cmd}
    ${cmd}
}

main "$@"
